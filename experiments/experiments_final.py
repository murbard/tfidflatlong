"""Final validation of the winning recipe (ridge + xyz target,
tf-idf letters-only min_df=5, SVD-100):

1. Strict CV: vectorizer and SVD fit on training folds only.
2. GBT ceiling on the same vocabulary (no SVD), for reference.
3. Per-country predictions (each from the fold where the country was
   held out) -> final map + CSV.
"""
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold

from experiments import load_data, haversine_km, to_xyz, from_xyz

LETTERS = r'(?u)\b[^\W\d_]{2,}\b'
VEC = dict(max_df=0.8, min_df=5, sublinear_tf=True, token_pattern=LETTERS)
SEEDS = (0, 1, 2)


def summarize(label, all_errs, t0):
    med = np.mean([np.median(e) for e in all_errs])
    mean = np.mean(all_errs)
    p90 = np.mean([np.percentile(e, 90) for e in all_errs])
    worst = np.mean([np.max(e) for e in all_errs])
    print(f'{label:45s} median={med:6.0f}  mean={mean:6.0f}  '
          f'p90={p90:6.0f}  max={worst:6.0f}  [{time.time() - t0:.0f}s]',
          flush=True)


def strict_cv(texts, latlon, svd_dim=100, model='ridge'):
    """Vectorizer + SVD fit on train folds only. Returns per-seed errors
    and (for seed 0) out-of-fold predicted lat/lon."""
    y = to_xyz(latlon)
    all_errs, oof_pred = [], np.zeros_like(latlon)
    for seed in SEEDS:
        errs = np.zeros(len(latlon))
        for tr, te in KFold(10, shuffle=True, random_state=seed).split(latlon):
            vec = TfidfVectorizer(**VEC)
            Xtr = vec.fit_transform([texts[i] for i in tr])
            Xte = vec.transform([texts[i] for i in te])
            if svd_dim:
                svd = TruncatedSVD(n_components=svd_dim, random_state=0)
                Xtr, Xte = svd.fit_transform(Xtr), svd.transform(Xte)
            else:
                Xtr, Xte = Xtr.toarray(), Xte.toarray()
            if model == 'ridge':
                m = RidgeCV(alphas=np.logspace(-3, 4, 30))
            else:
                m = MultiOutputRegressor(
                    HistGradientBoostingRegressor(
                        max_iter=300, learning_rate=0.1, min_samples_leaf=5),
                    n_jobs=3)
            m.fit(Xtr, y[tr])
            pred_ll = from_xyz(m.predict(Xte))
            errs[te] = haversine_km(latlon[te], pred_ll)
            if seed == 0:
                oof_pred[te] = pred_ll
        all_errs.append(errs)
        if model == 'gbt':
            break  # one seed is enough for the ceiling reference
    return np.array(all_errs), oof_pred


def plot_map(names, latlon, pred, errs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    img = plt.imread('files/Equirectangular-projection-topographic-world.jpg')
    fig, ax = plt.subplots(figsize=(24, 12))
    ax.imshow(img, extent=[-180, 180, -90, 90])
    for (lat, lon), (plat, plon), name in zip(latlon, pred, names):
        ax.plot([lon, plon], [lat, plat], 'k-', lw=0.6, alpha=0.6)
        ax.scatter(lon, lat, c='blue', s=12, zorder=3)
        ax.scatter(plon, plat, c='red', s=12, zorder=3)
    ax.set_title('Country centroid prediction from Wikipedia TF-IDF '
                 '(linear ridge, out-of-fold predictions; '
                 f'median error {np.median(errs):.0f} km)')
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    plt.tight_layout()
    plt.savefig('Country_Centroid_Prediction.png', dpi=120)
    print('wrote Country_Centroid_Prediction.png', flush=True)


if __name__ == '__main__':
    names, texts, latlon = load_data()
    print(f'{len(names)} countries/territories (Antarctica excluded)',
          flush=True)

    t0 = time.time()
    errs_svd, _ = strict_cv(texts, latlon, svd_dim=100, model='ridge')
    summarize('STRICT ridge + svd100', errs_svd, t0)

    t0 = time.time()
    errs, oof = strict_cv(texts, latlon, svd_dim=None, model='ridge')
    summarize('STRICT winner: ridge, no svd', errs, t0)

    t0 = time.time()
    errs_gbt, _ = strict_cv(texts, latlon, svd_dim=None, model='gbt')
    summarize('STRICT gbt ceiling (1 seed)', errs_gbt, t0)

    # final artifacts from seed-0 out-of-fold predictions of the winner
    e0 = errs[0]
    with open('final_predictions.csv', 'w') as f:
        f.write('country,lat,lon,pred_lat,pred_lon,err_km\n')
        for i in np.argsort(-e0):
            f.write(f'"{names[i]}",{latlon[i][0]:.2f},{latlon[i][1]:.2f},'
                    f'{oof[i][0]:.2f},{oof[i][1]:.2f},{e0[i]:.0f}\n')
    print('wrote final_predictions.csv', flush=True)
    print('\nWorst 10 (strict winner, seed 0):', flush=True)
    for i in np.argsort(-e0)[:10]:
        print(f'  {names[i]:40s} {e0[i]:6.0f} km', flush=True)
    plot_map(names, latlon, oof, e0)
