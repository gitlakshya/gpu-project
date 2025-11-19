import numpy as np
from numba import cuda, uint8, int32
from encrypt.key_expansion import key_expansion_device
from encrypt.AddRoundkey import add_round_key
from encrypt.sub_bytes_ import sub_bytes
from encrypt.shift_rows_ import shift_rows
from encrypt.mix_columns_ import mix_columns
from encrypt.round_ import round_function


# AES parameters for AES-128
NR_ROUNDS = 10           # AES-128 has 10 rounds
EXPANDED_KEY_SIZE = 16 * (NR_ROUNDS + 1)  # 176 bytes for AES-128

@cuda.jit
def aes_private_sharedlut(state,cipherKey,state_length,grcon,gsbox,gmul2,gmul3):
    """
    state         : uint8 array (global) - input / output buffer
    cipherKey     : uint8 array (global) - 16 bytes AES-128 key
    state_length  : int32 (global) - total bytes in state
    grcon, gsbox, gmul2, gmul3 : global uint8 arrays (256 each)
    """
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bdx= cuda.blockDim.x

    index = (tx +bdx*bx)*16  #Each thread processes 16 bytes

    #shared look-up tables (256 bytes each)
    s_rcon = cuda.shared.array((256,),dtype=uint8)
    s_sbox = cuda.shared.array((256,),dtype=uint8)
    s_mul2 = cuda.shared.array((256,),dtype=uint8)
    s_mul3 = cuda.shared.array((256,),dtype=uint8)

    #Load the global LUTs to the shared memory
    #If the block size < 256; thread 0 copies  entire table
    if bdx<256:
        if tx==0:
            for i in range(256):
                s_rcon[i] = grcon[i]
                s_sbox[i] = gsbox[i]
                s_mul2[i] = gmul2[i]
                s_mul3[i] = gmul3[i]

    else:
        #Each of the first 256 thread will copy one entry
        if tx<256:
            s_rcon[tx] = grcon[tx]
            s_sbox[tx] = gsbox[tx]
            s_mul2[tx] = gmul2[tx]
            s_mul3[tx] = gmul3[tx]    

    cuda.syncthreads()

    #shared expanded key
    s_expanded = cuda.shared.array((EXPANDED_KEY_SIZE,),dtype=uint8)


    #only thread 0 of the blocks expands the key
    if tx ==0:
        key_expansion_device(cipherKey,s_expanded,s_rcon,s_sbox)
    
    #Prepare  local storage (per -thread)
    state_local = cuda.local.array(16, uint8)

    #Load the block in to  local state if in range
    if index+16 < state_length:
        for i in range(16):
            state_local [i]=state[index+i]
    else:
        pass
        #can handle later if needed

    cuda.syncthreads()

    #If the block is valid lets perform the AES operation
    if index + 16 < state_length:
        #Intial Add round key
        #expanded key  round 0  starts at offset 0
        add_round_key(state_local,s_expanded[0:16]) # slicing produces view-like behavior in numba

        #Middle Rounds
        offset =16
        for r in range(1,NR_ROUNDS):
            #Call the round function with reference to round key at offset
            round_function(state_local,s_expanded[offset:offset+16],s_sbox,s_mul2,s_mul3)
            offset+=16

        #Final round: SubBytes, Shift Rows, AddRound Key (no Mix Columns)
        sub_bytes(state_local,s_sbox)
        shift_rows(state_local)
        add_round_key(state_local,s_expanded[NR_ROUNDS*16:NR_ROUNDS*16+16])

    cuda.syncthreads()

    # Write back
    if index + 16 <= state_length:
        for i in range(16):
            state[index + i] = state_local[i]
