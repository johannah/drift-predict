import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from glob import glob
import os
import sys
import pandas as pd
from utils import DATA_DIR
import pickle
import numpy as np

from IPython import embed; 

search = os.path.join(DATA_DIR, 'results', '*', 'result_distances.pkl')
print('searching', search)
result_files = sorted(glob(search))
print('found %s files'%len(result_files))
result_total = []
use_results = []
for rf in result_files:
    rr = pickle.load(open(rf, 'rb'))
    results =  []
    spot_names = []
    n_spots = len(rr.items())
    print(n_spots, rf)
    if n_spots < 50:
        print(n_spots, 'not enough spots', rf)
        break
    for key, item in rr.items():
        spot_names.append(key)
        item_valid = item[~np.isnan(item.sum(1)), :3]
        best_err = np.min(np.sum(item_valid, 1))
        results.append(best_err)
    print(rf, item.shape)
    use_results.append(rf)
    result_total.append(np.mean(results))

ss = [(os.path.split(os.path.split(y)[0])[1],x) for y, x in sorted(zip(use_results, result_total), key=lambda pair: pair[1])]
for s in ss:
    print(s)

