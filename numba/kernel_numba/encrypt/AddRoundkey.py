import numpy as np
from numba import cuda, uint8

#Add round key (Device function)

@cuda.jit(device=True)
def add_round_key(block,roundkey):
    block[0]  ^= roundkey[0]
    block[1]  ^= roundkey[1]
    block[2]  ^= roundkey[2]
    block[3]  ^= roundkey[3]
    block[4]  ^= roundkey[4]
    block[5]  ^= roundkey[5]
    block[6]  ^= roundkey[6]
    block[7]  ^= roundkey[7]
    block[8]  ^= roundkey[8]
    block[9]  ^= roundkey[9]
    block[10]  ^= roundkey[10]
    block[11]  ^= roundkey[11]
    block[12]  ^= roundkey[12]
    block[13]  ^= roundkey[13]
    block[14]  ^= roundkey[14]
    block[15]  ^= roundkey[15]


# -----------------------------------------
# Kernel to test AddRoundKey
# Equivalent to AddRoundKeyTest in CUDA
# -----------------------------------------
@cuda.jit
def AddRoundKeyTest(message, roundkey, length):
    idx = (cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x) * 16

    if idx + 16 <= length:
        # Create local 16-byte storage
        local_block = cuda.local.array(16, uint8)

        # Load block into local memory
        for i in range(16):
            local_block[i] = message[idx + i]

        # Apply AddRoundKey
        add_round_key(local_block, roundkey)

        # Store result back
        for i in range(16):
            message[idx + i] = local_block[i]

# -----------------------------------------
# Example usage
# -----------------------------------------
if __name__ == "__main__":
    # 32 bytes → 2 AES blocks
    msg = np.array([i for i in range(32)], dtype=np.uint8)

    # 16-byte fake round key
    rkey = np.array([0xAA for _ in range(16)], dtype=np.uint8)

    d_msg  = cuda.to_device(msg)
    d_rkey = cuda.to_device(rkey)

    threads = 4
    blocks = 1

    AddRoundKeyTest[blocks, threads](d_msg, d_rkey, np.int32(len(msg)))

    result = d_msg.copy_to_host()
    print("Output:", result)