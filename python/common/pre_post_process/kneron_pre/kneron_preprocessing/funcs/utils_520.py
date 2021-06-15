import math

def round_up_16(num):
    return ((num + (16 - 1)) & ~(16 - 1))

def round_up_n(num, n):
    if (num > 0):
        temp = float(num) / n
        return math.ceil(temp) * n
    else:
        return -math.ceil(float(-num) / n) * n

def cal_img_row_offset(crop_num, pad_num, start_row, out_row, orig_row):

    scaled_img_row = int(out_row - (pad_num[1] + pad_num[3]))
    if ((start_row - pad_num[1]) > 0):
        img_str_row = int((start_row - pad_num[1]))
    else:
        img_str_row = 0
    valid_row = int(orig_row - (crop_num[1] + crop_num[3]))
    img_str_row = int(valid_row * img_str_row / scaled_img_row)
    return int(img_str_row + crop_num[1])

def get_pad_num(pad_num_orig, left, up, right, bottom):
    pad_num = [0]*4
    for i in range(0,4):
        pad_num[i] = pad_num_orig[i]

    if not (left):
        pad_num[0] = 0
    if not (up):
        pad_num[1] = 0
    if not (right):
        pad_num[2] = 0
    if not (bottom):
        pad_num[3] = 0

    return pad_num

def get_byte_per_pixel(raw_fmt):
    if raw_fmt.lower() in ['RGB888', 'rgb888', 'RGB', 'rgb888']:
        return 4
    elif raw_fmt.lower() in ['YUV', 'yuv', 'YUV422', 'yuv422']:
        return 2
    elif raw_fmt.lower() in ['RGB565', 'rgb565']:
        return 2
    elif raw_fmt.lower() in ['NIR888', 'nir888', 'NIR', 'nir']:
        return 1
    else:
        return -1