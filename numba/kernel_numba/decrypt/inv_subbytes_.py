from numba import cuda, uint8

@cuda.jit(device=True)
def inv_subbytes(block, invsbox):
    for i in range(16):
        block[i]= invsbox[block[i]]


