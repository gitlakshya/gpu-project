import numpy as np
from numba import cuda

LUT_IN_SHARED = False   # Change to True if you want sbox passed per-block

@cuda.jit(device=True)
def sub_bytes(block,sbox):
    for i in range(16):
        block[i]=sbox[block[i]]


