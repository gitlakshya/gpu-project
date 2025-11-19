from numba import cuda, uint8

@cuda.jit(device=True)
def mix_columns(block,mul2,mul3):
    #temp buffer (16 bytes)
    temp = cuda.local.array(16,uint8)

    #process 4 columns
    for col in range(4):
        a0=block[4*col +0]
        a1=block[4*col +1]
        a2=block[4*col +2]
        a3=block[4*col +3]

        #Same logic as original cuda
        temp[4*col+0]=mul2[a0] ^ mul3[a1] ^ a2 ^ a3
        temp[4*col+1]=a0 ^ mul2[a1] ^ mul3[a2] ^ a3
        temp[4*col+2]=a0 ^ a1 ^ mul2[a2] ^ mul3[a3]
        temp[4*col+3]=mul3[a0] ^ a1^ a2 ^ mul2[a3]

    #Copy the result back to the block
    for i in range(16):
        block[i]=temp[i]