import numpy as np
from PIL import Image

def twos_complement(value):
    value = int(value)
    # msb = (value & 0x8000) * (1/np.power(2, 15))
    msb = (value & 0x8000) >> 15
    if msb == 1:
        if (((~value) & 0xFFFF) + 1) >= 0xFFFF:
            result = ((~value) & 0xFFFF)
        else:
            result = (((~value) & 0xFFFF) + 1)
        result = result * (-1)
    else:
        result = value

    return result


def twos_complement_pix(value):
    h, _ = value.shape
    for i in range(h):
        value[i, 0] = twos_complement(value[i, 0])

    return value

def clip(value, mini, maxi):
    if value < mini:
        result = mini
    elif value > maxi:
        result = maxi
    else:
        result = value

    return result

def clip_pix(value, mini, maxi):
    h, _ = value.shape
    for i in range(h):
        value[i, 0] = clip(value[i, 0], mini, maxi)

    return value