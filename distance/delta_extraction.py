import numpy as np
import librosa as lib 


def deltas(y, max_len):
    d  = lib.feature.delta(y, width= max_len, axis = 0, order = 1)
    dd = lib.feature.delta(y, width= max_len, axis = 0, order = 2)
    return d, dd




