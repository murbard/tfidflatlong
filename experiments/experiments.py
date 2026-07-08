"""Systematic evaluation of tf-idf -> lat/long prediction.

Compares target encodings (raw lat/lon vs 3D unit vector), vectorizer
settings, and models, using repeated k-fold CV with great-circle error.
"""
import os
import csv
import re
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import MultiTaskElasticNetCV, RidgeCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics.pairwise import cosine_similarity

R_EARTH = 6371.0


COORD_LINE = re.compile(r'^.*[°′″].*$', re.MULTILINE)
DECIMAL_PAIR = re.compile(r'-?\d{1,3}\.\d+;\s*-?\d{1,3}\.\d+')


def strip_coordinates(text):
    """Remove Wikipedia coordinate strings (footer and inline) — they
    literally contain the answer."""
    text = COORD_LINE.sub(' ', text)
    text = DECIMAL_PAIR.sub(' ', text)
    return text


def load_data():
    coords = {}
    with open(os.path.join('files', 'countries.csv'), newline='') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            name, lat, lon = row[2], float(row[1]), float(row[0])
            coords[name] = (lat, lon)
    texts, latlon, names = [], [], []
    coords.pop('Antarctica', None)  # not a country; centroid is a polygon artifact
    for name, (lat, lon) in coords.items():
        path = os.path.join('files', f'{name}.html')
        if not os.path.exists(path):
            print(f'missing: {name}')
            continue
        with open(path) as f:
            texts.append(strip_coordinates(f.read()))
        latlon.append((lat, lon))
        names.append(name)
    return names, texts, np.array(latlon)


def haversine_km(a, b):
    """a, b: (..., 2) arrays of (lat, lon) in degrees."""
    lat1, lon1 = np.radians(a[..., 0]), np.radians(a[..., 1])
    lat2, lon2 = np.radians(b[..., 0]), np.radians(b[..., 1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def to_xyz(latlon):
    lat, lon = np.radians(latlon[:, 0]), np.radians(latlon[:, 1])
    return np.stack([np.cos(lat) * np.cos(lon),
                     np.cos(lat) * np.sin(lon),
                     np.sin(lat)], axis=1)


def from_xyz(xyz):
    n = np.linalg.norm(xyz, axis=1, keepdims=True)
    xyz = xyz / np.maximum(n, 1e-12)
    lat = np.degrees(np.arcsin(np.clip(xyz[:, 2], -1, 1)))
    lon = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    return np.stack([lat, lon], axis=1)


def make_model(kind):
    if kind == 'enet':
        return MultiTaskElasticNetCV(l1_ratio=np.linspace(0.05, 0.95, 7),
                                     max_iter=20000, cv=5, n_jobs=-1)
    if kind == 'ridge':
        return RidgeCV(alphas=np.logspace(-3, 4, 30))
    if kind == 'krr_cos':
        return GridSearchCV(
            KernelRidge(kernel='cosine'),
            {'alpha': np.logspace(-4, 1, 12)}, cv=10, n_jobs=-1)
    if kind == 'knn_cos':
        return GridSearchCV(
            KNeighborsRegressor(metric='cosine', weights='distance'),
            {'n_neighbors': [1, 2, 3, 5, 8, 12]}, cv=10, n_jobs=-1)
    if kind == 'gbt':
        return MultiOutputRegressor(
            HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1,
                                          min_samples_leaf=5), n_jobs=3)
    raise ValueError(kind)


def run(names, texts, latlon, vec_kwargs, model_kind, target,
        n_splits=10, seed=0):
    """Returns per-country great-circle errors from k-fold CV."""
    vec = TfidfVectorizer(**vec_kwargs)
    X = vec.fit_transform(texts)  # fit on all: vocab choice only, unsupervised
    y = to_xyz(latlon) if target == 'xyz' else latlon.copy()
    errs = np.zeros(len(names))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in kf.split(X):
        model = make_model(model_kind)
        Xtr, Xte = X[tr], X[te]
        if model_kind in ('enet', 'ridge', 'gbt'):
            Xtr, Xte = Xtr.toarray(), Xte.toarray()
        model.fit(Xtr, y[tr])
        pred = model.predict(Xte)
        pred_ll = from_xyz(pred) if target == 'xyz' else pred
        errs[te] = haversine_km(latlon[te], pred_ll)
    return errs, X.shape[1]


def report(label, errs, nfeat, t0):
    print(f'{label:55s} vocab={nfeat:6d}  median={np.median(errs):7.0f} km  '
          f'mean={np.mean(errs):7.0f} km  p90={np.percentile(errs, 90):7.0f} km  '
          f'max={np.max(errs):7.0f} km  [{time.time() - t0:.0f}s]',
          flush=True)


if __name__ == '__main__':
    names, texts, latlon = load_data()
    print(f'{len(names)} countries/territories loaded')

    LETTERS = r'(?u)\b[^\W\d_]{2,}\b'  # exclude all-numeric tokens
    VEC_ORIG = dict(max_df=0.8, min_df=0.1)
    VEC_WIDE = dict(max_df=0.8, min_df=2, sublinear_tf=True,
                    token_pattern=LETTERS)

    configs = [
        # label, vectorizer, model, target
        ('baseline: orig vec + enet + raw latlon', VEC_ORIG, 'enet', 'latlon'),
        ('orig vec + enet + xyz', VEC_ORIG, 'enet', 'xyz'),
        ('orig vec + ridge + xyz', VEC_ORIG, 'ridge', 'xyz'),
        ('wide vec + ridge + xyz', VEC_WIDE, 'ridge', 'xyz'),
        ('wide vec + ridge + raw latlon', VEC_WIDE, 'ridge', 'latlon'),
        ('wide vec + krr_cos + xyz', VEC_WIDE, 'krr_cos', 'xyz'),
        ('wide vec + knn_cos + xyz', VEC_WIDE, 'knn_cos', 'xyz'),
        ('wide vec + enet + xyz', VEC_WIDE, 'enet', 'xyz'),
        ('orig vec + gbt + xyz (ceiling ref)', VEC_ORIG, 'gbt', 'xyz'),
        ('wide vec + gbt + xyz (ceiling ref)', VEC_WIDE, 'gbt', 'xyz'),
    ]
    for label, vec_kwargs, model_kind, target in configs:
        t0 = time.time()
        errs, nfeat = run(names, texts, latlon, vec_kwargs, model_kind, target)
        report(label, errs, nfeat, t0)
