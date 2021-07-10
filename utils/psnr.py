import math
import numpy as np
def rmse_score(true, pred):
    score = math.sqrt(np.mean((true-pred)**2))
    return score

def calc_psnr(sr, hr, rgb_range=255):
    diff = (sr - hr) / rgb_range
    mse = diff.pow(2).mean()
    return -10 * math.log10(mse)