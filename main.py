"""Predict country centroids (lat, lon) from the tf-idf of their Wikipedia page.

Winning recipe (see experiments*.py / results_round*.txt for the search):
  - strip literal coordinate strings from the pages (they contain the answer)
  - tf-idf: letters-only tokens, max_df=0.8, min_df=5, sublinear tf
  - linear ridge predicting a 3D point, projected back onto the unit sphere
    (avoids the longitude-wraparound penalty of regressing raw lat/lon)

Outputs:
  - strict 10-fold CV metrics (vectorizer refit on training folds only)
  - final_predictions.csv                : out-of-fold predictions, worst first
  - Country_Centroid_Prediction.png      : map of out-of-fold predictions
  - Country_Centroid_Fullfit.png         : map of a model fit on the whole
                                           sample with the CV-optimal alpha
"""
import csv
import os
import re
import time

import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold

R_EARTH = 6371.0
LETTERS = r'(?u)\b[^\W\d_]{2,}\b'  # exclude tokens containing digits
VEC = dict(max_df=0.8, min_df=5, sublinear_tf=True, token_pattern=LETTERS)
ALPHAS = np.logspace(-3, 4, 30)
SEEDS = (0, 1, 2)

# Wikipedia blocks the default python-requests user agent
HEADERS = {'User-Agent': 'tfidflatlong/1.0 (research; contact via github.com/murbard)'}

COORD_LINE = re.compile(r'^.*[°′″].*$', re.MULTILINE)
DECIMAL_PAIR = re.compile(r'-?\d{1,3}\.\d+;\s*-?\d{1,3}\.\d+')


# ---------------------------------------------------------------- data

def fetch_coordinates():
    """Country polygon barycenters from gavinr/world-countries-centroids."""
    url = ('https://raw.githubusercontent.com/gavinr/'
           'world-countries-centroids/master/dist/countries.csv')
    path = os.path.join('files', 'countries.csv')
    if not os.path.exists(path):
        with open(path, 'wb') as f:
            f.write(requests.get(url, headers=HEADERS).content)
    coords = {}
    with open(path, newline='') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            name, lat, lon = row[2], float(row[1]), float(row[0])
            coords[name] = (lat, lon)
    # not a country; its "centroid" is a pole-wrapping polygon artifact
    coords.pop('Antarctica', None)
    return coords


def fetch_map_image():
    path = os.path.join('files', 'Equirectangular-projection-topographic-world.jpg')
    url = ('https://upload.wikimedia.org/wikipedia/commons/3/3e/'
           'Equirectangular-projection-topographic-world.jpg')
    if not os.path.exists(path):
        with open(path, 'wb') as f:
            f.write(requests.get(url, headers=HEADERS).content)
    return path


def fetch_descriptions(coords):
    """Plain text of each country's Wikipedia page, cached under files/."""
    name_mapper = {
        'Saba': 'Saba_(island)',
        'Saint Martin': 'Saint_Martin_(island)',
        'Congo': 'Republic_of_the_Congo',
        'Congo DRC': 'Democratic_Republic_of_the_Congo',
        'Georgia': 'Georgia_(country)',
        'Canarias': 'Canary_Islands',
    }
    desc = {}
    for name in coords:
        path = os.path.join('files', f'{name}.html')
        if not os.path.exists(path):
            url_name = name_mapper.get(name, name)
            html = requests.get(f'https://en.wikipedia.org/wiki/{url_name}',
                                headers=HEADERS).text
            with open(path, 'w') as f:
                f.write(BeautifulSoup(html, 'html.parser').get_text())
        with open(path) as f:
            desc[name] = f.read()
    return desc


def strip_coordinates(text):
    """Remove Wikipedia coordinate strings — they literally contain the answer."""
    return DECIMAL_PAIR.sub(' ', COORD_LINE.sub(' ', text))


# ---------------------------------------------------------------- geometry

def haversine_km(a, b):
    """a, b: (..., 2) arrays of (lat, lon) in degrees."""
    lat1, lon1 = np.radians(a[..., 0]), np.radians(a[..., 1])
    lat2, lon2 = np.radians(b[..., 0]), np.radians(b[..., 1])
    h = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def to_xyz(latlon):
    lat, lon = np.radians(latlon[:, 0]), np.radians(latlon[:, 1])
    return np.stack([np.cos(lat) * np.cos(lon),
                     np.cos(lat) * np.sin(lon),
                     np.sin(lat)], axis=1)


def from_xyz(xyz):
    xyz = xyz / np.maximum(np.linalg.norm(xyz, axis=1, keepdims=True), 1e-12)
    lat = np.degrees(np.arcsin(np.clip(xyz[:, 2], -1, 1)))
    lon = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    return np.stack([lat, lon], axis=1)


# ---------------------------------------------------------------- model

def cross_validate(texts, latlon):
    """Strict 10-fold CV (vectorizer refit per fold), repeated over SEEDS.

    Returns per-seed error arrays, seed-0 out-of-fold predictions, and the
    ridge alphas selected in each fold.
    """
    y = to_xyz(latlon)
    all_errs, alphas = [], []
    oof_pred = np.zeros_like(latlon)
    for seed in SEEDS:
        errs = np.zeros(len(latlon))
        for tr, te in KFold(10, shuffle=True, random_state=seed).split(latlon):
            vec = TfidfVectorizer(**VEC)
            Xtr = vec.fit_transform([texts[i] for i in tr]).toarray()
            Xte = vec.transform([texts[i] for i in te]).toarray()
            model = RidgeCV(alphas=ALPHAS)
            model.fit(Xtr, y[tr])
            alphas.append(model.alpha_)
            pred_ll = from_xyz(model.predict(Xte))
            errs[te] = haversine_km(latlon[te], pred_ll)
            if seed == 0:
                oof_pred[te] = pred_ll
        all_errs.append(errs)
    return np.array(all_errs), oof_pred, np.array(alphas)


def fit_full(texts, latlon, alpha):
    """Fit on the whole sample with a fixed alpha; return in-sample preds."""
    X = TfidfVectorizer(**VEC).fit_transform(texts).toarray()
    model = Ridge(alpha=alpha)
    model.fit(X, to_xyz(latlon))
    return from_xyz(model.predict(X))


# ---------------------------------------------------------------- output

def summarize(label, all_errs):
    med = np.mean([np.median(e) for e in all_errs])
    mean = np.mean(all_errs)
    p90 = np.mean([np.percentile(e, 90) for e in all_errs])
    worst = np.mean([np.max(e) for e in all_errs])
    print(f'{label:45s} median={med:6.0f}  mean={mean:6.0f}  '
          f'p90={p90:6.0f}  max={worst:6.0f}  (km)', flush=True)


def wrapped_segments(lon1, lat1, lon2, lat2):
    """Line segments from (lon1,lat1) to (lon2,lat2) going the short way
    around, split at the antimeridian when needed."""
    d = lon2 - lon1
    if abs(d) <= 180:
        return [([lon1, lon2], [lat1, lat2])]
    lon2s = lon2 - 360 if d > 0 else lon2 + 360
    edge = -180 if lon2s < -180 else 180
    t = (edge - lon1) / (lon2s - lon1)
    latc = lat1 + t * (lat2 - lat1)
    return [([lon1, edge], [lat1, latc]), ([-edge, lon2], [latc, lat2])]


def plot_map(fname, title, names, latlon, pred, errs, n_highlight=12):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors, cm, patheffects

    img = plt.imread(fetch_map_image())
    fig, ax = plt.subplots(figsize=(28, 14))
    ax.imshow(img, extent=[-180, 180, -90, 90], alpha=0.85)

    norm = mcolors.PowerNorm(0.5, vmin=0, vmax=max(errs.max(), 1.0))
    cmap = plt.get_cmap('YlOrRd')
    halo = [patheffects.withStroke(linewidth=1.6, foreground='white', alpha=0.85)]
    worst = set(np.argsort(-errs)[:n_highlight]) if errs.max() > 100 else set()

    for i, ((lat, lon), (plat, plon)) in enumerate(zip(latlon, pred)):
        color = cmap(norm(errs[i]))
        for xs, ys in wrapped_segments(lon, lat, plon, plat):
            ax.plot(xs, ys, '-', color=color, lw=1.3, alpha=0.9, zorder=2)
        ax.scatter(lon, lat, c='#1040c0', s=16, zorder=4,
                   edgecolors='white', linewidths=0.4)
        ax.scatter(plon, plat, c='#d02020', s=16, zorder=4,
                   edgecolors='white', linewidths=0.4)
        if i in worst:
            ax.annotate(f'{names[i]} ({errs[i]:.0f} km)', (lon, lat),
                        textcoords='offset points', xytext=(4, 4),
                        fontsize=8, fontweight='bold', color='#701010',
                        path_effects=halo, zorder=5)
        else:
            ax.annotate(names[i], (lon, lat),
                        textcoords='offset points', xytext=(3, 3),
                        fontsize=4.5, color='#202030',
                        path_effects=halo, zorder=5)

    ax.scatter([], [], c='#1040c0', s=30, label='actual centroid')
    ax.scatter([], [], c='#d02020', s=30, label='predicted')
    ax.legend(loc='lower left', fontsize=11, framealpha=0.9)
    stats = (f'median {np.median(errs):>5.0f} km\n'
             f'mean   {np.mean(errs):>5.0f} km\n'
             f'p90    {np.percentile(errs, 90):>5.0f} km\n'
             f'max    {np.max(errs):>5.0f} km')
    ax.text(0.011, 0.135, stats, transform=ax.transAxes, fontsize=11,
            family='monospace', va='bottom',
            bbox=dict(facecolor='white', alpha=0.85, boxstyle='round,pad=0.4'))

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, fraction=0.018, pad=0.005,
                 label='great-circle error (km)')
    ax.set_title(title, fontsize=15)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print(f'wrote {fname}', flush=True)


if __name__ == '__main__':
    coords = fetch_coordinates()
    desc = fetch_descriptions(coords)
    names = list(coords)
    texts = [strip_coordinates(desc[n]) for n in names]
    latlon = np.array([coords[n] for n in names])
    print(f'{len(names)} countries/territories (Antarctica excluded)', flush=True)

    t0 = time.time()
    all_errs, oof, alphas = cross_validate(texts, latlon)
    summarize('strict 10-fold CV (ridge, tf-idf, 3D target)', all_errs)
    print(f'alphas selected across folds: median={np.median(alphas):g} '
          f'(range {alphas.min():g}–{alphas.max():g})  '
          f'[{time.time() - t0:.0f}s]', flush=True)

    e0 = all_errs[0]
    with open('final_predictions.csv', 'w') as f:
        f.write('country,lat,lon,pred_lat,pred_lon,err_km\n')
        for i in np.argsort(-e0):
            f.write(f'"{names[i]}",{latlon[i][0]:.2f},{latlon[i][1]:.2f},'
                    f'{oof[i][0]:.2f},{oof[i][1]:.2f},{e0[i]:.0f}\n')
    print('wrote final_predictions.csv', flush=True)

    plot_map('Country_Centroid_Prediction.png',
             'Country centroid prediction from Wikipedia tf-idf '
             '(linear ridge, out-of-fold predictions; '
             f'median error {np.median(e0):.0f} km)',
             names, latlon, oof, e0)

    # in-sample fit on the whole corpus, regularized with the CV-optimal alpha
    alpha = np.median(alphas)
    pred_full = fit_full(texts, latlon, alpha)
    errs_full = haversine_km(latlon, pred_full)
    summarize(f'full-sample fit (alpha={alpha:g}, in-sample)', [errs_full])
    plot_map('Country_Centroid_Fullfit.png',
             'Country centroid prediction from Wikipedia tf-idf '
             f'(linear ridge fit on the whole sample, alpha={alpha:g}; '
             f'in-sample median error {np.median(errs_full):.0f} km)',
             names, latlon, pred_full, errs_full)
