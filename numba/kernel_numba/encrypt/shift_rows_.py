from numba import cuda, uint8

@cuda.jit(device=True)
def shift_rows(block):
       # --------------------------------------
    # ROW 0: No shift
    # --------------------------------------

    # --------------------------------------
    # ROW 1: Shift left by 1
    # indices: 1, 5, 9, 13
    # --------------------------------------
    temp = block[1]
    block[1]=block[5]
    block[5]=block[9]
    block[9]=block[13]
    block[13]=temp

    # --------------------------------------
    # ROW 2: Shift left by 2
    # indices: 2 ↔ 10, 6 ↔ 14
    # --------------------------------------

    temp = block[2]
    block[2]=block[10]
    block[10]=temp

    temp = block[6]
    block[6]=block[14]
    block[14]=temp

    # --------------------------------------
    # ROW 3: Shift left by 3 (or right by 1)
    # indices: 3, 7, 11, 15
    # --------------------------------------

    temp = block[15]
    block[15]=block[11]
    block[11]=block[7]
    block[7]=block[3]
    block[3]=temp
    