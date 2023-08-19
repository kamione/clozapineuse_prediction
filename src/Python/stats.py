import numpy as np


def mssd(data):
    mssd = np.power(np.ediff1d(data), 2).mean()
    return mssd
