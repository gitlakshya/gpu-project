from numba import cuda,uint8
from encrypt.mix_columns_ import mix_columns

@cuda.jit(device=True)
def inv_mixcols(block, mul2,mul3): # mul3 is unused but kept for API consistency
    u= uint8(0)
    v= uint8(0)

    for col in range(4):
        #u =mul2[mul2[(block[col*4]^block[col*4+2])]]
        t1 = block[4*col]^block[4*col+2]
        u =mul2 [mul2[t1]]

        #v = mul2[mul2[(block[col*4+1]^block[col*4+3])]]
        t2 = block[col*4+1]^block[col*4+3]
        v =mul2[mul2[t2]]

        #Apply the XORs
        block [4* col]^= u 
        block [4* col + 1]^=v
        block [4* col + 2]^=u
        block [4* col + 3]^=v
    mix_columns(block, mul2, mul3)   