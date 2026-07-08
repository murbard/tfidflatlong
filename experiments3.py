"""Round 3: fine sweep of the two knobs that mattered (min_df, SVD dim)
and their combination. Ridge + xyz throughout, 3x10-fold repeated CV.
"""
import time
from experiments import load_data
from experiments2 import run_ridge, report, BASE

if __name__ == '__main__':
    names, texts, latlon = load_data()

    configs = []
    for mdf in (3, 5, 8, 12, 20):
        configs.append((f'min_df={mdf}', {**BASE, 'min_df': mdf}, None))
    for svd in (50, 75, 100, 150):
        configs.append((f'min_df=2 svd={svd}', BASE, svd))
    for svd in (50, 75, 100, 150):
        configs.append((f'min_df=5 svd={svd}', {**BASE, 'min_df': 5}, svd))
    configs.append(('min_df=8 svd=100', {**BASE, 'min_df': 8}, 100))

    for label, vk, svd in configs:
        t0 = time.time()
        all_errs, nfeat = run_ridge(texts, latlon, vk, svd)
        report(label, all_errs, nfeat, t0)
