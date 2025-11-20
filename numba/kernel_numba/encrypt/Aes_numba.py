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
def aes_private_sharedlut(state, cipherKey, state_length, grcon, gsbox, gmul2, gmul3):
    """
    state         : 1D uint8 device array (input/output)
    cipherKey     : 1D uint8 device array (length 16)
    state_length  : total bytes in state (int)
    grcon, gsbox, gmul2, gmul3 : 1D uint8 device arrays (256 each)
    """

    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bdx = cuda.blockDim.x

    index = (tx + bdx * bx) * 16  # byte index of this thread's AES block

    # -------------------------
    # Shared LUTs (256 bytes)
    # -------------------------
    s_rcon = cuda.shared.array(256, uint8)
    s_sbox = cuda.shared.array(256, uint8)
    s_mul2 = cuda.shared.array(256, uint8)
    s_mul3 = cuda.shared.array(256, uint8)

    # Load LUTs from global -> shared
    if bdx < 256:
        if tx == 0:
            for i in range(256):
                s_rcon[i] = grcon[i]
                s_sbox[i] = gsbox[i]
                s_mul2[i] = gmul2[i]
                s_mul3[i] = gmul3[i]
    else:
        if tx < 256:
            s_rcon[tx] = grcon[tx]
            s_sbox[tx] = gsbox[tx]
            s_mul2[tx] = gmul2[tx]
            s_mul3[tx] = gmul3[tx]

    cuda.syncthreads()

    # -------------------------
    # Shared expanded key (176 bytes)
    # -------------------------
    s_expanded = cuda.shared.array(EXPANDED_KEY_SIZE, uint8)

    # Only thread 0 performs key expansion (per block)
    if tx == 0:
        key_expansion_device(cipherKey, s_expanded, s_rcon, s_sbox)

    cuda.syncthreads()

    # -------------------------
    # Local (per-thread) state buffer
    # -------------------------
    state_local = cuda.local.array(16, uint8)

    # Load block into local buffer (only if within range)
    if index + 16 <= state_length:
        for i in range(16):
            state_local[i] = state[index + i]
    else:
        return  # out-of-range thread does nothing

    # -------------------------
    # AES encryption using shared expanded key + shared LUTs
    # -------------------------
    # Initial AddRoundKey (round 0) — copy round key to local buffer first
    round_key_local = cuda.local.array(16, uint8)
    # offset for round 0
    offset = 0
    for j in range(16):
        round_key_local[j] = s_expanded[offset + j]
    add_round_key(state_local, round_key_local)

    # Middle rounds
    offset = 16
    for r in range(1, NR_ROUNDS):
        # copy round key for this round into local buffer
        for j in range(16):
            round_key_local[j] = s_expanded[offset + j]
        # perform one AES round (SubBytes, ShiftRows, MixColumns, AddRoundKey)
        round_function(state_local, round_key_local, s_sbox, s_mul2, s_mul3)
        offset += 16

    # Final round (no MixColumns)
    sub_bytes(state_local, s_sbox)
    shift_rows(state_local)
    # final round key (round NR_ROUNDS)
    for j in range(16):
        round_key_local[j] = s_expanded[NR_ROUNDS * 16 + j]
    add_round_key(state_local, round_key_local)

    # -------------------------
    # Write back to global memory
    # -------------------------
    for i in range(16):
        state[index + i] = state_local[i]
