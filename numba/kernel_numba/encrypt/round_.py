from numba import cuda
from encrypt.shift_rows_ import shift_rows
from encrypt.sub_bytes_ import sub_bytes
from encrypt.AddRoundkey import add_round_key
from encrypt.mix_columns_ import mix_columns

@cuda.jit(device = True)
def round_function(block,roundkey,sbox,mul2,mul3):
    #Apply Subbytes
    sub_bytes(block,sbox)
    #Apply ShiftRows
    shift_rows(block)
    #Apply MixColumns
    mix_columns(block,mul2,mul3)
    #Apply AddRoundKey
    add_round_key(block,roundkey)
