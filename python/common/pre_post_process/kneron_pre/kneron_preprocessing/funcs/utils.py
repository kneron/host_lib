import numpy as np
from PIL import Image
import struct

def pad_square_to_4(x_start, x_end, y_start, y_end):
    w_int = x_end - x_start 
    h_int = y_end - y_start
    pad = w_int - h_int
    if pad > 0:
        pad_s = (pad >> 1) &(~3)
        pad_e = pad - pad_s
        y_start -= pad_s
        y_end += pad_e
    else:#//pad <=0
        pad_s = -(((pad) >> 1) &(~3))
        pad_e = (-pad) - pad_s
        x_start -= pad_s
        x_end += pad_e
    return x_start, x_end, y_start, y_end

def str_fill(value):
    if len(value) == 1:
        value = "0" + value
    elif len(value) == 0:
        value = "00"

    return value

def clip_ary(value):
    list_v = []
    for i in range(len(value)):
        v = value[i] % 256
        list_v.append(v)

    return list_v
    
def str2bool(v):
    if isinstance(v,bool):
        return v
    return v.lower() in ('TRUE', 'True', 'true', '1', 'T', 't', 'Y', 'YES', 'y', 'yes')


def str2int(s):
    if s == "":
        s = 0
    s = int(s)
    return s

def str2float(s):
    if s == "":
        s = 0
    s = float(s)
    return s

def clip(value, mini, maxi):
    if value < mini:
        result = mini
    elif value > maxi:
        result = maxi
    else:
        result = value

    return result


def clip_ary(value):
    list_v = []
    for i in range(len(value)):
        v = value[i] % 256
        list_v.append(v)

    return list_v


def signed_rounding(value, bit):
    if value < 0:
        value = value - (1 << (bit - 1))
    else:
        value = value + (1 << (bit - 1))

    return value

def hex_loader(data_folder,**kwargs):
    format_mode = kwargs['raw_img_fmt']
    src_h = kwargs['img_in_height']
    src_w = kwargs['img_in_width']

    if format_mode in ['YUV444', 'yuv444', 'YCBCR444', 'YCbCr444', 'ycbcr444']:
        output = hex_yuv444(data_folder,src_h,src_w)
    elif format_mode in ['RGB565', 'rgb565']:
        output = hex_rgb565(data_folder,src_h,src_w)
    elif format_mode in ['YUV422', 'yuv422', 'YCBCR422', 'YCbCr422', 'ycbcr422']:
        output = hex_yuv422(data_folder,src_h,src_w)

    return output

def hex_rgb565(hex_folder,src_h,src_w):
    pix_per_line = 8
    byte_per_line = 16

    f = open(hex_folder)
    pixel_r = []
    pixel_g = []
    pixel_b = []

    # Ignore the first line
    f.readline()
    input_line = int((src_h * src_w)/pix_per_line)
    for i in range(input_line):
        readline = f.readline()
        for j in range(int(byte_per_line/2)-1, -1, -1):
            data1 = int(readline[(j * 4 + 0):(j * 4 + 2)], 16)
            data0 = int(readline[(j * 4 + 2):(j * 4 + 4)], 16)
            r = ((data1 & 0xf8) >> 3)
            g = (((data0 & 0xe0) >> 5) + ((data1 & 0x7) << 3))
            b = (data0 & 0x1f)
            pixel_r.append(r)
            pixel_g.append(g)
            pixel_b.append(b)

    ary_r = np.array(pixel_r, dtype=np.uint8)
    ary_g = np.array(pixel_g, dtype=np.uint8)
    ary_b = np.array(pixel_b, dtype=np.uint8)
    output = np.concatenate((ary_r[:, None], ary_g[:, None], ary_b[:, None]), axis=1)
    output = output.reshape((src_h, src_w, 3))

    return output

def hex_yuv444(hex_folder,src_h,src_w):
    pix_per_line = 4
    byte_per_line = 16

    f = open(hex_folder)
    byte0 = []
    byte1 = []
    byte2 = []
    byte3 = []

    # Ignore the first line
    f.readline()
    input_line = int((src_h * src_w)/pix_per_line)
    for i in range(input_line):
        readline = f.readline()
        for j in range(byte_per_line-1, -1, -1):
            data = int(readline[(j*2):(j*2+2)], 16)
            if (j+1) % 4 == 0:
                byte0.append(data)
            elif (j+2) % 4 == 0:
                byte1.append(data)
            elif (j+3) % 4 == 0:
                byte2.append(data)
            elif (j+4) % 4 == 0:
                byte3.append(data)
    # ary_a = np.array(byte0, dtype=np.uint8)
    ary_v = np.array(byte1, dtype=np.uint8)
    ary_u = np.array(byte2, dtype=np.uint8)
    ary_y = np.array(byte3, dtype=np.uint8)
    output = np.concatenate((ary_y[:, None], ary_u[:, None], ary_v[:, None]), axis=1)
    output = output.reshape((src_h, src_w, 3))

    return output

def hex_yuv422(hex_folder,src_h,src_w):
    pix_per_line = 8
    byte_per_line = 16
    f = open(hex_folder)
    pixel_y = []
    pixel_u = []
    pixel_v = []

    # Ignore the first line
    f.readline()
    input_line = int((src_h * src_w)/pix_per_line)
    for i in range(input_line):
        readline = f.readline()
        for j in range(int(byte_per_line/4)-1, -1, -1):
            data3 = int(readline[(j * 8 + 0):(j * 8 + 2)], 16)
            data2 = int(readline[(j * 8 + 2):(j * 8 + 4)], 16)
            data1 = int(readline[(j * 8 + 4):(j * 8 + 6)], 16)
            data0 = int(readline[(j * 8 + 6):(j * 8 + 8)], 16)
            pixel_y.append(data3)
            pixel_y.append(data1)
            pixel_u.append(data2)
            pixel_u.append(data2)
            pixel_v.append(data0)
            pixel_v.append(data0)

    ary_y = np.array(pixel_y, dtype=np.uint8)
    ary_u = np.array(pixel_u, dtype=np.uint8)
    ary_v = np.array(pixel_v, dtype=np.uint8)
    output = np.concatenate((ary_y[:, None], ary_u[:, None], ary_v[:, None]), axis=1)
    output = output.reshape((src_h, src_w, 3))

    return output

def bin_loader(data_folder,**kwargs):
    format_mode = kwargs['raw_img_fmt']
    src_h = kwargs['img_in_height']
    src_w = kwargs['img_in_width']
    if format_mode in ['YUV','yuv','YUV444', 'yuv444', 'YCBCR','YCbCr','ycbcr','YCBCR444', 'YCbCr444', 'ycbcr444']:
        output = bin_yuv444(data_folder,src_h,src_w)
    elif format_mode in ['RGB565', 'rgb565']:
        output = bin_rgb565(data_folder,src_h,src_w)
    elif format_mode in ['NIR', 'nir','NIR888', 'nir888']:
        output = bin_nir(data_folder,src_h,src_w)
    elif format_mode in ['YUV422', 'yuv422', 'YCBCR422', 'YCbCr422', 'ycbcr422']:
        output = bin_yuv422(data_folder,src_h,src_w)
    elif format_mode in ['RGB888','rgb888']:
        output = np.fromfile(data_folder, dtype='uint8')
        output = output.reshape(src_h,src_w,3)
    elif format_mode in ['RGBA8888','rgba8888', 'RGBA' , 'rgba']:
        output_temp = np.fromfile(data_folder, dtype='uint8')
        output_temp = output_temp.reshape(src_h,src_w,4)
        output = output_temp[:,:,0:3]

    return output

def bin_yuv444(in_img_path,src_h,src_w):
    # load bin
    struct_fmt = '1B' 
    struct_len = struct.calcsize(struct_fmt)
    struct_unpack = struct.Struct(struct_fmt).unpack_from
    
    row = src_h
    col = src_w
    pixels = row*col

    raw = []
    with open(in_img_path, "rb") as f:
        while True:
            data = f.read(struct_len)
            if not data: break
            s = struct_unpack(data)
            raw.append(s[0])
    

    raw = raw[:pixels*4]

    #
    output = np.zeros((pixels * 3), dtype=np.uint8)
    cnt = 0
    for i in range(0, pixels*4, 4):
        #Y
        output[cnt] = raw[i+3]
        #U
        cnt += 1
        output[cnt] = raw[i+2]
        #V
        cnt += 1
        output[cnt] = raw[i+1]

        cnt += 1          

    output = output.reshape((src_h,src_w,3))
    return output
    
def bin_yuv422(in_img_path,src_h,src_w):
    # load bin
    struct_fmt = '1B' 
    struct_len = struct.calcsize(struct_fmt)
    struct_unpack = struct.Struct(struct_fmt).unpack_from
    
    row = src_h
    col = src_w
    pixels = row*col

    raw = []
    with open(in_img_path, "rb") as f:
        while True:
            data = f.read(struct_len)
            if not data: break
            s = struct_unpack(data)
            raw.append(s[0])
    

    raw = raw[:pixels*2]

    #
    output = np.zeros((pixels * 3), dtype=np.uint8)
    cnt = 0
    for i in range(0, pixels*2, 4):
        #Y0
        output[cnt] = raw[i+3]
        #U0
        cnt += 1
        output[cnt] = raw[i+2]
        #V0
        cnt += 1
        output[cnt] = raw[i]
        #Y1
        cnt += 1
        output[cnt] = raw[i+1]
        #U1
        cnt += 1
        output[cnt] = raw[i+2]
        #V1
        cnt += 1
        output[cnt] = raw[i]

        cnt += 1          

    output = output.reshape((src_h,src_w,3))
    return output

def bin_rgb565(in_img_path,src_h,src_w):
    # load bin
    struct_fmt = '1B' 
    struct_len = struct.calcsize(struct_fmt)
    struct_unpack = struct.Struct(struct_fmt).unpack_from
    
    row = src_h
    col = src_w
    pixels = row*col

    rgba565 = []
    with open(in_img_path, "rb") as f:
        while True:
            data = f.read(struct_len)
            if not data: break
            s = struct_unpack(data)
            rgba565.append(s[0])
    

    rgba565 = rgba565[:pixels*2]

    # rgb565_bin to numpy_array
    output = np.zeros((pixels * 3), dtype=np.uint8)
    cnt = 0
    for i in range(0, pixels*2, 2):
        temp = rgba565[i]
        temp2 = rgba565[i+1]
        #R-5
        output[cnt] = (temp2 >>3)
        
        #G-6
        cnt += 1
        output[cnt] = ((temp & 0xe0) >> 5) + ((temp2 & 0x07) << 3)
        
        #B-5
        cnt += 1
        output[cnt] = (temp & 0x1f)

        cnt += 1          

    output = output.reshape((src_h,src_w,3))
    return output

def bin_nir(in_img_path,src_h,src_w):
    # load bin
    struct_fmt = '1B' 
    struct_len = struct.calcsize(struct_fmt)
    struct_unpack = struct.Struct(struct_fmt).unpack_from

    nir = []
    with open(in_img_path, "rb") as f:
        while True:
            data = f.read(struct_len)
            if not data: break
            s = struct_unpack(data)
            nir.append(s[0])
            
    nir = nir[:src_h*src_w]
    pixels = len(nir)
    # nir_bin to numpy_array
    output = np.zeros((len(nir) * 3), dtype=np.uint8)
    for i in range(0, pixels):
        output[i*3]=nir[i]
        output[i*3+1]=nir[i]
        output[i*3+2]=nir[i]

    output = output.reshape((src_h,src_w,3))
    return output
