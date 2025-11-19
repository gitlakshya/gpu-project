from numba import cuda, uint8
from encrypt.AddRoundkey  import add_round_key
from inv_shift_rows_ import inv_shift_rows
from inv_subbytes_ import inv_subbytes

@cuda.jit(device=True)
def inv_final_round(block,roundkey,invsbox):
    #Invert shift rows
    inv_shift_rows(block)

    # Inverse SubBytes
    inv_subbytes(block, invsbox)

    # AddRoundKey
    add_round_key(block,roundkey)

