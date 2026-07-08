"""Round 2: chip at the error. Fix ridge + xyz target (round-1 winner),
sweep vectorizer settings, text preprocessing, and SVD; repeated k-fold
to separate real gains from CV noise. Dumps per-country errors for the
best config.
"""
import re
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold

from experiments import (load_data, haversine_km, to_xyz, from_xyz)

SEEDS = (0, 1, 2)


def truncate_tail(text):
    """Drop everything from 'Retrieved from' on (categories metadata)."""
    i = text.rfind('Retrieved from')
    return text[:i] if i > 0 else text


def drop_references(text):
    """Cut at the References section heading if present."""
    m = re.search(r'\nReferences\n', text)
    return text[:m.start()] if m else text


def run_ridge(texts, latlon, vec_kwargs, svd_dim=None):
    vec = TfidfVectorizer(**vec_kwargs)
    X = vec.fit_transform(texts)
    if svd_dim:
        X = TruncatedSVD(n_components=svd_dim, random_state=0).fit_transform(X)
    else:
        X = X.toarray()
    y = to_xyz(latlon)
    all_errs = []
    for seed in SEEDS:
        errs = np.zeros(len(latlon))
        for tr, te in KFold(10, shuffle=True, random_state=seed).split(X):
            model = RidgeCV(alphas=np.logspace(-3, 4, 30))
            model.fit(X[tr], y[tr])
            errs[te] = haversine_km(latlon[te], from_xyz(model.predict(X[te])))
        all_errs.append(errs)
    return np.array(all_errs), X.shape[1]


def report(label, all_errs, nfeat, t0):
    med = np.mean([np.median(e) for e in all_errs])
    mean = np.mean(all_errs)
    p90 = np.mean([np.percentile(e, 90) for e in all_errs])
    worst = np.mean([np.max(e) for e in all_errs])
    print(f'{label:58s} d={nfeat:6d}  median={med:6.0f}  mean={mean:6.0f}  '
          f'p90={p90:6.0f}  max={worst:6.0f}  [{time.time() - t0:.0f}s]',
          flush=True)
    return med


LETTERS = r'(?u)\b[^\W\d_]{2,}\b'
BASE = dict(max_df=0.8, min_df=2, sublinear_tf=True, token_pattern=LETTERS)

if __name__ == '__main__':
    names, texts, latlon = load_data()
    texts_trunc = [truncate_tail(t) for t in texts]
    texts_noref = [drop_references(t) for t in texts_trunc]
    print(f'{len(names)} countries; mean len full/trunc/noref: '
          f'{np.mean([len(t) for t in texts]):.0f}/'
          f'{np.mean([len(t) for t in texts_trunc]):.0f}/'
          f'{np.mean([len(t) for t in texts_noref]):.0f}', flush=True)

    configs = [
        ('base (round-1 winner)', texts, BASE, None),
        ('trunc tail (no categories)', texts_trunc, BASE, None),
        ('no references section', texts_noref, BASE, None),
        ('min_df=1', texts, {**BASE, 'min_df': 1}, None),
        ('min_df=5', texts, {**BASE, 'min_df': 5}, None),
        ('max_df=0.5', texts, {**BASE, 'max_df': 0.5}, None),
        ('max_df=0.9', texts, {**BASE, 'max_df': 0.9}, None),
        ('no sublinear_tf', texts, {**BASE, 'sublinear_tf': False}, None),
        ('binary tf', texts, {**BASE, 'binary': True}, None),
        ('stopwords english', texts, {**BASE, 'stop_words': 'english'}, None),
        ('bigrams, max 100k', texts,
         {**BASE, 'ngram_range': (1, 2), 'max_features': 100000}, None),
        ('svd 100', texts, BASE, 100),
        ('svd 200', texts, BASE, 200),
        ('svd 300', texts, BASE, 300),
    ]
    results = {}
    for label, txts, vk, svd in configs:
        t0 = time.time()
        all_errs, nfeat = run_ridge(txts, latlon, vk, svd)
        results[label] = report(label, all_errs, nfeat, t0)

    # per-country errors for the base config, for outlier analysis
    all_errs, _ = run_ridge(texts, latlon, BASE)
    mean_err = all_errs.mean(axis=0)
    order = np.argsort(-mean_err)
    print('\nWorst 20 countries (base config, mean over seeds):', flush=True)
    for i in order[:20]:
        print(f'  {names[i]:45s} {mean_err[i]:6.0f} km '
              f'(centroid {latlon[i][0]:6.1f}, {latlon[i][1]:7.1f})', flush=True)
    with open('per_country_errors.csv', 'w') as f:
        f.write('country,lat,lon,mean_err_km\n')
        for i in order:
            f.write(f'"{names[i]}",{latlon[i][0]},{latlon[i][1]},{mean_err[i]:.1f}\n')
    print('\nwrote per_country_errors.csv', flush=True)
