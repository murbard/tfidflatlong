"""Gradient-boosting ceiling reference (nonlinear, one config per vocab)."""
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold

from experiments import load_data, haversine_km, to_xyz, from_xyz

LETTERS = r'(?u)\b[^\W\d_]{2,}\b'

if __name__ == '__main__':
    names, texts, latlon = load_data()
    y = to_xyz(latlon)
    for label, vk in [
        ('orig vec + gbt + xyz', dict(max_df=0.8, min_df=0.1)),
        ('wide vec + gbt + xyz', dict(max_df=0.8, min_df=2, sublinear_tf=True,
                                      token_pattern=LETTERS)),
    ]:
        t0 = time.time()
        X = TfidfVectorizer(**vk).fit_transform(texts).toarray()
        errs = np.zeros(len(names))
        for tr, te in KFold(10, shuffle=True, random_state=0).split(X):
            model = MultiOutputRegressor(
                HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1,
                                              min_samples_leaf=5), n_jobs=3)
            model.fit(X[tr], y[tr])
            errs[te] = haversine_km(latlon[te], from_xyz(model.predict(X[te])))
        print(f'{label:55s} vocab={X.shape[1]:6d}  '
              f'median={np.median(errs):7.0f} km  mean={np.mean(errs):7.0f} km  '
              f'p90={np.percentile(errs, 90):7.0f} km  max={np.max(errs):7.0f} km'
              f'  [{time.time() - t0:.0f}s]', flush=True)
