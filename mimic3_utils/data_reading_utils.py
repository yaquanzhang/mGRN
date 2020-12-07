from joblib import Parallel, delayed
import time


def preprocess_chunk_parallel(data, ts, discretizer, normalizer=None, n_worders = 20):
    print('discretizer')
    beg = time.time()
    # data = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
    # an object of discretizer is copied to all workers.
    # statistics will not work
    data = Parallel(n_jobs=n_worders)(delayed(discretizer.transform)(X, end=t) for (X, t) in zip(data, ts))
    data = [x[0] for x in data]
    print("time spent:", time.time() - beg)
    if normalizer is not None:
        print('normalizaer')
        beg = time.time()
        data = [normalizer.transform(X) for X in data]
        print("time spent:", time.time() - beg)
    return data

