import numpy as np
from numba import cuda, uint8, int32
NR_ROUNDS = 10
EXPANDED_KEY_SIZE = 16 * (NR_ROUNDS + 1)

# RotByte: [a b c d] → [b c d a]
@cuda.jit(device=True)
def RotByte(word):
    temp=word[0]
    word[0]=word[1]
    word[1]=word[2]
    word[2]=word[3]
    word[3]=temp
# SubByte: applies S-box to 4 bytes
@cuda.jit(device=True)
def SubByte(word,sbox):
    word[0] = sbox[word[0]]
    word[1] = sbox[word[1]]
    word[2] = sbox[word[2]]
    word[3] = sbox[word[3]]


# -----------------------------------
# KeyExpansion
# AES-128 → output = 176 bytes
# -----------------------------------
@cuda.jit(device=True)
def key_expansion_device(cipher_key,expanded_key,rcon,sbox):

    #Copy the initial 16 bytes
    for i in range(16):
        expanded_key[i]=cipher_key[i]
    
    #expand the remaining  160 bytes
    temp = cuda.local.array(4, uint8)
    
    #i index grows  4 bytes at a time
    for i in range (16, EXPANDED_KEY_SIZE, 4):

        #temp = previous word
        for j in range(4):
            temp[j]=expanded_key[i-4+j]

        #Every Fourth word -> special transform 
        if i % 16 ==0:
            RotByte(temp)
            SubByte(temp,sbox)
            temp[0]^=rcon[i//16]   # rcon[1], rcon[2], ...

        #W[i] = W[i-16] XOR temp
        for j in range(4):
            expanded_key[i+j]=expanded_key[i-16 +j ] ^  temp[j]

# -----------------------------------
# Test Kernel (equivalent to original)
# -----------------------------------
@cuda.jit
def KeyExpansionTest(cipher_key, expanded_key, rcon, sbox):
    # Only 1 thread needed
    if cuda.threadIdx.x == 0:
        key_expansion_device(cipher_key, expanded_key, rcon, sbox)