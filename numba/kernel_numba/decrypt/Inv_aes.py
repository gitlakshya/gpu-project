from numba import cuda, uint8
from encrypt.AddRoundkey import add_round_key
from encrypt.key_expansion import key_expansion_device
from inv_final_round_ import inv_final_round
from inv_round_ import inv_round
NR_ROUNDS = 10
EXPANDED_KEY_SIZE = 16 * (NR_ROUNDS+1)

@cuda.jit
def inv_AES (state,cipher_key,statelength,grcon,gsbox,ginvsbox,gmul2,gmul3):
    #compute block index
    idx = (cuda.threadIdx.x + cuda.BlockDim.x*cuda.BlockIdx.x)*16

    #Shared memory lookuptable
    s_rcon   = cuda.shared.array(256,uint8)
    s_sbox   = cuda.shared.array(256,uint8)
    s_invbox = cuda.shared.array(256,uint8)
    s_mul2   = cuda.shared.array(256,uint8)
    s_mul3   = cuda.shared.array(256,uint8)

    #load LUts in to the shared memory
    tx  = cuda.threadIdX.x
    bdx = cuda.blockDim.x

    if bdx <256:
        #thread 0 loads  everything
        if tx==0:
            for i in range(256):
                s_rcon[i]=grcon[i]                 
                s_sbox[i]=gsbox[i]
                s_invbox[i]=ginvsbox[i]
                s_mul2[i]=gmul2[i]
                s_mul3[i]=gmul3[i]
        else:
            #first 256 threads does the work  cooperatively
            if tx<256:
                 s_rcon[tx]=grcon[tx]                 
                 s_sbox[tx]=gsbox[tx]
                 s_invbox[tx]=ginvsbox[tx]
                 s_mul2[tx]=gmul2[tx]
                 s_mul3[tx]=gmul3[tx]

        cuda.syncthreads()

    #shared expanded key
    s_expanded = cuda.shared.array(EXPANDED_KEY_SIZE,uint8)

    # only thread 0 expands the key

    if tx==0:
        key_expansion_device(cipher_key,s_expanded,s_rcon,s_sbox)

    cuda.syncthreads()


    #Load the State bloc in to the private memory
    stateLocal=cuda.local.array(16,uint8)

    if idx +16 <statelength:
        for i in range(16):
            stateLocal[i]=state[idx +i]

    #Decrypt the block
    if idx+16 <= statelength:

        #start with last round key
        add_round_key(stateLocal,stateLocal,s_expanded[16*NR_ROUNDS:(16*NR_ROUNDS+1)])

        #middle inverse rounds
        for r in range(1,NR_ROUNDS):
            key_offset=16*(NR_ROUNDS-r)
            inv_round(stateLocal,s_expanded[key_offset:key_offset+16],
                      s_invbox,
                      s_mul2,
                      s_mul3)
        #final inverse round (no Invmix columns)
        inv_final_round(stateLocal,s_expanded[0:16],s_invbox)

    cuda.syncthreads()

    # ======== Write Back Result ========
    if idx + 16 <= statelength:
        for i in range(16):
            state[idx + i] = stateLocal[i]