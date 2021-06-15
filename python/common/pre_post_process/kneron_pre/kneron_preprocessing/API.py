# -*- coding: utf-8 -*-

import numpy as np
import os
from .funcs.utils import str2int, str2bool
from . import Flow

flow = Flow()
flow.set_numerical_type('floating')
flow_520 = Flow()
flow_520.set_numerical_type('520')
flow_720 = Flow()
flow_720.set_numerical_type('720')

DEFAULT = None
default = {
    'crop':{
        'align_w_to_4':False
        },
    'resize':{
        'type':'bilinear',
        'calculate_ratio_using_CSim':False
        }
}

def set_default_as_520():
    """
    Set some default parameter as 520 setting

    crop.align_w_to_4 = True
    crop.pad_square_to_4 = True
    resize.type = 'fixed_520'
    resize.calculate_ratio_using_CSim = True
    """
    global default
    default['crop']['align_w_to_4'] = True
    default['resize']['type'] = 'fixed_520'
    default['resize']['calculate_ratio_using_CSim'] = True
    return

def set_default_as_floating():
    """
    Set some default parameter as floating setting

    crop.align_w_to_4 = False
    crop.pad_square_to_4 = False
    resize.type = 'bilinear'
    resize.calculate_ratio_using_CSim = False
    """
    global default
    default['crop']['align_w_to_4'] = False
    default['resize']['type'] = 'bilinear'
    default['resize']['calculate_ratio_using_CSim'] = False
    pass

def print_info_on():
    """
    turn print infomation on.
    """
    flow.set_print_info(True)
    flow_520.set_print_info(True)

def print_info_off():
    """
    turn print infomation off.
    """
    flow.set_print_info(False)
    flow_520.set_print_info(False)

def load_image(image):
    """
    load_image function
    load load_image and output as rgb888 format np.array

    Args:
        image: [np.array/str], can be np.array or image file path

    Returns:
        out: [np.array], rgb888 format

    Examples:
    """
    image = flow.load_image(image, is_raw = False)
    return image

def load_bin(image, fmt=None, size=None):
    """
    load_bin function
    load bin file and output as rgb888 format np.array

    Args:
        image: [str], bin file path
        fmt: [str], "rgb888" / "rgb565" / "nir"
        size: [tuble], (image_w, image_h)

    Returns:
        out: [np.array], rgb888 format

    Examples:
        >>> image_data = kneron_preprocessing.API.load_bin(image,'rgb565',(raw_w,raw_h))
    """    
    assert isinstance(size, tuple)
    assert isinstance(fmt, str)
    # assert (fmt.lower() in ['rgb888', "rgb565" , "nir",'RGB888', "RGB565" , "NIR", 'NIR888', 'nir888'])

    image = flow.load_image(image, is_raw = True, raw_img_type='bin', raw_img_fmt = fmt, img_in_width = size[0], img_in_height = size[1])
    flow.set_color_conversion(source_format=fmt, out_format = 'rgb888')
    image,_ = flow.funcs['color'](image)
    return image

def load_hex(file, fmt=None, size=None):
    """
    load_hex function
    load hex file and output as rgb888 format np.array

    Args:
        image: [str], hex file path
        fmt: [str], "rgb888" / "yuv444" / "ycbcr444" / "yuv422" / "ycbcr422" / "rgb565"
        size: [tuble], (image_w, image_h)

    Returns:
        out: [np.array], rgb888 format

    Examples:
        >>> image_data = kneron_preprocessing.API.load_hex(image,'rgb565',(raw_w,raw_h))
    """  
    assert isinstance(size, tuple)
    assert isinstance(fmt, str)
    assert (fmt.lower() in ['rgb888',"yuv444" , "ycbcr444" , "yuv422" , "ycbcr422" , "rgb565"])

    image = flow.load_image(file, is_raw = True, raw_img_type='hex', raw_img_fmt = fmt, img_in_width = size[0], img_in_height = size[1])
    flow.set_color_conversion(source_format=fmt, out_format = 'rgb888')
    image,_ = flow.funcs['color'](image)
    return image

def dump_image(image, output=None, file_fmt='txt',image_fmt='rgb888',order=0):
    """
    dump_image function

    dump txt, bin or hex, default is txt
    image format as following format: RGB888, RGBA8888, RGB565, NIR, YUV444, YCbCr444, YUV422, YCbCr422, default is RGB888

    Args:
        image: [np.array/str], can be np.array or image file path
        output: [str], dump file path
        file_fmt: [str], "bin" / "txt" / "hex", set dump file format, default is txt
        image_fmt: [str], RGB888 / RGBA8888 / RGB565 / NIR / YUV444 / YCbCr444 / YUV422 / YCbCr422, default is RGB888

    Examples:
        >>> kneron_preprocessing.API.dump_image(image_data,out_path,fmt='bin')
    """
    if isinstance(image, str):
        image = load_image(image)

    assert isinstance(image, np.ndarray)
    if output is None:
        return

    flow.set_output_setting(is_dump=False, dump_format=file_fmt, image_format=image_fmt ,output_file=output)
    flow.dump_image(image)
    return

def convert(image, out_fmt = 'RGB888', source_fmt = 'RGB888'):
    """
    color convert

    Args:
        image: [np.array], input
        out_fmt: [str], "rgb888" / "rgba8888" / "rgb565" / "yuv" / "ycbcr" / "yuv422" / "ycbcr422"
        source_fmt: [str], "rgb888" / "rgba8888" / "rgb565" / "yuv" / "ycbcr" / "yuv422" / "ycbcr422"

    Returns:
        out: [np.array]

    Examples:

    """  
    flow.set_color_conversion(source_format = source_fmt, out_format=out_fmt, simulation=False)
    image,_ = flow.funcs['color'](image)
    return image

def get_crop_range(box,align_w_to_4=DEFAULT, pad_square_to_4=False,rounding_type=0):
    """
    get exact crop box according different setting

    Args:
        box: [tuble], (x1, y1, x2, y2)
        align_w_to_4: [bool], crop length in w direction align to 4 or not, default False
        pad_square_to_4: [bool], pad to square(align 4) or not, default False
        rounding_type: [int], 0-> x1,y1 take floor, x2,y2 take ceil; 1->all take rounding 

    Returns:
        out: [tuble,4], (crop_x1, crop_y1, crop_x2, crop_y2) 

    Examples:
        >>> image_data = kneron_preprocessing.API.get_crop_range((272,145,461,341), align_w_to_4=True, pad_square_to_4=True)
        (272, 145, 460, 341)
    """  
    if box is None:
        return (0,0,0,0)
    if align_w_to_4 is None:
        align_w_to_4 = default['crop']['align_w_to_4']

    flow.set_crop(type='specific', start_x=box[0],start_y=box[1],end_x=box[2],end_y=box[3], align_w_to_4=align_w_to_4, pad_square_to_4=pad_square_to_4,rounding_type=rounding_type)
    image = np.zeros((1,1,3)).astype('uint8')
    _,info = flow.funcs['crop'](image)
    
    return info['box']

def crop(image, box=None, align_w_to_4=DEFAULT, pad_square_to_4=False,rounding_type=0 ,info_out = {}):
    """
    crop function

    specific crop range by box

    Args:
        image: [np.array], input
        box: [tuble], (x1, y1, x2, y2)
        align_w_to_4: [bool], crop length in w direction align to 4 or not, default False
        pad_square_to_4: [bool], pad to square(align 4) or not, default False
        rounding_type: [int], 0-> x1,y1 take floor, x2,y2 take ceil; 1->all take rounding 
        info_out: [dic], save the final crop box into info_out['box']

    Returns:
        out: [np.array] 

    Examples:
        >>> info = {}
        >>> image_data = kneron_preprocessing.API.crop(image_data,(272,145,461,341), align_w_to_4=True, info_out=info)
        >>> info['box']
        (272, 145, 460, 341)

        >>> info = {}
        >>> image_data = kneron_preprocessing.API.crop(image_data,(272,145,461,341), pad_square_to_4=True, info_out=info)
        >>> info['box']
        (268, 145, 464, 341)
    """  
    assert isinstance(image, np.ndarray)
    if box is None:
        return image
    if align_w_to_4 is None:
        align_w_to_4 = default['crop']['align_w_to_4']

    flow.set_crop(type='specific', start_x=box[0],start_y=box[1],end_x=box[2],end_y=box[3], align_w_to_4=align_w_to_4, pad_square_to_4=pad_square_to_4,rounding_type=rounding_type)
    image,info = flow.funcs['crop'](image)
    
    info_out['box'] = info['box']
    return image

def crop_center(image, range=None, align_w_to_4=DEFAULT, pad_square_to_4=False,rounding_type=0 ,info_out = {}):
    """
    crop function

    center crop by range

    Args:
        image: [np.array], input
        range: [tuble], (crop_w, crop_h)
        align_w_to_4: [bool], crop length in w direction align to 4 or not, default False
        pad_square_to_4: [bool], pad to square(align 4) or not, default False
        rounding_type: [int], 0-> x1,y1 take floor, x2,y2 take ceil; 1->all take rounding 
        info_out: [dic], save the final crop box into info_out['box']

    Returns:
        out: [np.array] 

    Examples:
        >>> info = {}
        >>> image_data = kneron_preprocessing.API.crop_center(image_data,(102,40), align_w_to_4=True,info_out=info)
        >>> info['box']
        (268, 220, 372, 260)

        >>> info = {}
        >>> image_data = kneron_preprocessing.API.crop_center(image_data,(102,40), pad_square_to_4=True, info_out=info)
        >>> info['box']
        (269, 192, 371, 294)
    """   
    assert isinstance(image, np.ndarray)
    if range is None:
        return image
    if align_w_to_4 is None:
        align_w_to_4 = default['crop']['align_w_to_4']

    flow.set_crop(type='center', crop_w=range[0],crop_h=range[1], align_w_to_4=align_w_to_4, pad_square_to_4=pad_square_to_4,rounding_type=rounding_type)
    image,info = flow.funcs['crop'](image)

    info_out['box'] = info['box']
    return image

def crop_corner(image, range=None, align_w_to_4=DEFAULT,pad_square_to_4=False,rounding_type=0 ,info_out = {}):
    """
    crop function

    corner crop by range

    Args:
        image: [np.array], input
        range: [tuble], (crop_w, crop_h)
        align_w_to_4: [bool], crop length in w direction align to 4 or not, default False
        pad_square_to_4: [bool], pad to square(align 4) or not, default False
        rounding_type: [int], 0-> x1,y1 take floor, x2,y2 take ceil; 1->all take rounding 
        info_out: [dic], save the final crop box into info_out['box']

    Returns:
        out: [np.array] 

    Examples:
        >>> info = {}
        >>> image_data = kneron_preprocessing.API.crop_corner(image_data,(102,40), align_w_to_4=True,info_out=info)
        >>> info['box']
        (0, 0, 104, 40)

        >>> info = {}
        >>> image_data = kneron_preprocessing.API.crop_corner(image_data,(102,40), pad_square_to_4=True,info_out=info)
        >>> info['box']
        (0, -28, 102, 74)
    """
    assert isinstance(image, np.ndarray)
    if range is None:
        return image
    if align_w_to_4 is None:
        align_w_to_4 = default['crop']['align_w_to_4']

    flow.set_crop(type='corner', crop_w=range[0],crop_h=range[1], align_w_to_4=align_w_to_4, pad_square_to_4=pad_square_to_4)
    image, info = flow.funcs['crop'](image)

    info_out['box'] = info['box']
    return image

def resize(image, size=None, keep_ratio = True, zoom = True, type=DEFAULT, calculate_ratio_using_CSim = DEFAULT, info_out = {}):
    """
    resize function

    resize type can be bilinear or bilicubic as floating type, fixed or fixed_520/fixed_720 as fixed type.
    fixed_520/fixed_720 type has add some function to simulate 520/720 bug.

    Args:
        image: [np.array], input
        size: [tuble], (input_w, input_h)
        keep_ratio: [bool], keep_ratio or not, default True
        zoom: [bool], enable resize can zoom image or not, default True
        type: [str], "bilinear" / "bilicubic" / "cv2" / "fixed" / "fixed_520" / "fixed_720"
        calculate_ratio_using_CSim: [bool], calculate the ratio and scale using Csim function and C float, default False
        info_out: [dic], save the final scale size(w,h) into info_out['size']

    Returns:
        out: [np.array] 

    Examples:
        >>> info = {}
        >>> image_data = kneron_preprocessing.API.resize(image_data,size=(56,56),type='fixed',info_out=info)
        >>> info_out['size']
        (54,56)
    """
    assert isinstance(image, np.ndarray)
    if size is None:
        return image
    if type is None:
        type = default['resize']['type']
    if calculate_ratio_using_CSim is None:
        calculate_ratio_using_CSim = default['resize']['calculate_ratio_using_CSim']

    flow.set_resize(resize_w = size[0], resize_h = size[1], type=type, keep_ratio=keep_ratio,zoom=zoom, calculate_ratio_using_CSim=calculate_ratio_using_CSim)
    image, info = flow.funcs['resize'](image)
    info_out['size'] = info['size']

    return image

def pad(image, pad_l=0, pad_r=0, pad_t=0, pad_b=0, pad_val=0):
    """
    pad function

    specific left, right, top and bottom pad size.

    Args:
        image[np.array]: input
        pad_l: [int], pad size from left, default 0
        pad_r: [int], pad size form right, default 0
        pad_t: [int], pad size from top, default 0
        pad_b: [int], pad size form bottom, default 0
        pad_val: [float], the value of pad, , default 0 

    Returns:
        out: [np.array] 

    Examples:
        >>> image_data = kneron_preprocessing.API.pad(image_data,20,40,20,40,-0.5)
    """
    assert isinstance(image, np.ndarray)

    flow.set_padding(type='specific',pad_l=pad_l,pad_r=pad_r,pad_t=pad_t,pad_b=pad_b,pad_val=pad_val)
    image, _ = flow.funcs['padding'](image)
    return image

def pad_center(image,size=None, pad_val=0):
    """
    pad function

    center pad with pad size.

    Args:
        image[np.array]: input
        size: [tuble], (padded_size_w, padded_size_h)
        pad_val: [float], the value of pad, , default 0 

    Returns:
        out: [np.array] 

    Examples:
        >>> image_data = kneron_preprocessing.API.pad_center(image_data,size=(56,56),pad_val=-0.5)
    """
    assert isinstance(image, np.ndarray)
    if size is None:
        return image
    assert ( (image.shape[0] <= size[1]) & (image.shape[1] <= size[0]) )

    flow.set_padding(type='center',padded_w=size[0],padded_h=size[1],pad_val=pad_val)
    image, _ = flow.funcs['padding'](image)
    return image

def pad_corner(image,size=None, pad_val=0):
    """
    pad function

    corner pad with pad size.

    Args:
        image[np.array]: input
        size: [tuble], (padded_size_w, padded_size_h)
        pad_val: [float], the value of pad, , default 0 

    Returns:
        out: [np.array] 

    Examples:
        >>> image_data = kneron_preprocessing.API.pad_corner(image_data,size=(56,56),pad_val=-0.5)
    """   
    assert isinstance(image, np.ndarray)
    if size is None:
        return image
    assert ( (image.shape[0] <= size[1]) & (image.shape[1] <= size[0]) )

    flow.set_padding(type='corner',padded_w=size[0],padded_h=size[1],pad_val=pad_val)
    image, _ = flow.funcs['padding'](image)
    return image

def norm(image,scale=256.,bias=-0.5, mean=None, std=None):
    """
    norm function
    
    x = (x/scale - bias)
    x[0,1,2] = x - mean[0,1,2]
    x[0,1,2] = x / std[0,1,2]

    Args:
        image: [np.array], input
        scale: [float], default = 256
        bias: [float], default = -0.5
        mean: [tuble,3], default = None
        std: [tuble,3], default = None

    Returns:
        out: [np.array] 

    Examples:
        >>> image_data = kneron_preprocessing.API.norm(image_data)
        >>> image_data = kneron_preprocessing.API.norm(image_data,mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    """  
    assert isinstance(image, np.ndarray)

    flow.set_normalize(type='specific',scale=scale,  bias=bias, mean=mean, std =std)
    image, _ = flow.funcs['normalize'](image)
    return image

def inproc_520(image,raw_fmt='rgb565',raw_size=None,npu_size=None, crop_box=None, pad_mode=0, norm='kneron', gray=False, rotate=0, radix=8, bit_width=8, round_w_to_16=True, NUM_BANK_LINE=32,BANK_ENTRY_CNT=512,MAX_IMG_PREPROC_ROW_NUM=511,MAX_IMG_PREPROC_COL_NUM=256):
    """
    inproc_520

    Args:
        image: [np.array], input
        crop_box: [tuble], (x1, y1, x2, y2), if None will skip crop
        pad_mode: [int], 0: pad 2 sides, 1: pad 1 side, 2: no pad. default = 0
        norm: [str], default = 'kneron'
        rotate: [int], 0 / 1 / 2 ,default = 0
        radix: [int], default = 8
        bit_width: [int], default = 8
        round_w_to_16: [bool], default = True
        gray: [bool], default = False

    Returns:
        out: [np.array] 

    Examples:
        >>> image_data = kneron_preprocessing.API.inproc_520(image_data,npu_size=(56,56),crop_box=(272,145,460,341),pad_mode=1)
    """  
    # assert isinstance(image, np.ndarray)

    if (not isinstance(image, np.ndarray)):
        flow_520.set_raw_img(is_raw_img='yes',raw_img_type = 'bin',raw_img_fmt=raw_fmt, img_in_width=raw_size[0], img_in_height=raw_size[1])
    else:
        flow_520.set_raw_img(is_raw_img='no')
        flow_520.set_color_conversion(source_format='rgb888')

    if npu_size is None:
        return image

    flow_520.set_model_size(w=npu_size[0],h=npu_size[1])

    ## Crop
    if crop_box != None:
        flow_520.set_crop(start_x=crop_box[0],start_y=crop_box[1],end_x=crop_box[2],end_y=crop_box[3])
        crop_fisrt = True
    else:
        crop_fisrt = False

    ## Color
    if gray:
        flow_520.set_color_conversion(out_format='l',simulation='no')
    else:
        flow_520.set_color_conversion(out_format='rgb888',simulation='no')

    ## Resize & Pad
    pad_mode = str2int(pad_mode)
    if (pad_mode == 0):
        pad_type = 'center'
        resize_keep_ratio = 'yes'
    elif (pad_mode == 1):
        pad_type = 'corner'
        resize_keep_ratio = 'yes'
    else:
        pad_type = 'center'
        resize_keep_ratio = 'no'
    
    flow_520.set_resize(keep_ratio=resize_keep_ratio)
    flow_520.set_padding(type=pad_type)

    ## Norm
    flow_520.set_normalize(type=norm)

    ## 520 inproc
    flow_520.set_520_setting(radix=radix,bit_width=bit_width,rotate=rotate,crop_fisrt=crop_fisrt,round_w_to_16=round_w_to_16,NUM_BANK_LINE=NUM_BANK_LINE,BANK_ENTRY_CNT=BANK_ENTRY_CNT,MAX_IMG_PREPROC_ROW_NUM=MAX_IMG_PREPROC_ROW_NUM,MAX_IMG_PREPROC_COL_NUM=MAX_IMG_PREPROC_COL_NUM)
    image_data, _ = flow_520.run_whole_process(image)

    return image_data

def inproc_720(image,raw_fmt='rgb565',raw_size=None,npu_size=None, crop_box=None, pad_mode=0, norm='kneron', gray=False):
    """
    inproc_720

    Args:
        image: [np.array], input
        crop_box: [tuble], (x1, y1, x2, y2), if None will skip crop
        pad_mode: [int], 0: pad 2 sides, 1: pad 1 side, 2: no pad. default = 0
        norm: [str], default = 'kneron'
        rotate: [int], 0 / 1 / 2 ,default = 0
        radix: [int], default = 8
        bit_width: [int], default = 8
        round_w_to_16: [bool], default = True
        gray: [bool], default = False

    Returns:
        out: [np.array] 

    Examples:
        >>> image_data = kneron_preprocessing.API.inproc_520(image_data,npu_size=(56,56),crop_box=(272,145,460,341),pad_mode=1)
    """  
    # assert isinstance(image, np.ndarray)

    if (not isinstance(image, np.ndarray)):
        flow_720.set_raw_img(is_raw_img='yes',raw_img_type = 'bin',raw_img_fmt=raw_fmt, img_in_width=raw_size[0], img_in_height=raw_size[1])
    else:
        flow_720.set_raw_img(is_raw_img='no')
        flow_720.set_color_conversion(source_format='rgb888')

    if npu_size is None:
        return image

    flow_720.set_model_size(w=npu_size[0],h=npu_size[1])

    ## Crop
    if crop_box != None:
        flow_720.set_crop(start_x=crop_box[0],start_y=crop_box[1],end_x=crop_box[2],end_y=crop_box[3])
        crop_fisrt = True
    else:
        crop_fisrt = False

    ## Color
    if gray:
        flow_720.set_color_conversion(out_format='l',simulation='no')
    else:
        flow_720.set_color_conversion(out_format='rgb888',simulation='no')

    ## Resize & Pad
    pad_mode = str2int(pad_mode)
    if (pad_mode == 0):
        pad_type = 'center'
        resize_keep_ratio = 'yes'
    elif (pad_mode == 1):
        pad_type = 'corner'
        resize_keep_ratio = 'yes'
    else:
        pad_type = 'center'
        resize_keep_ratio = 'no'
    
    flow_720.set_resize(keep_ratio=resize_keep_ratio)
    flow_720.set_padding(type=pad_type)

    ## 720 inproc
    # flow_720.set_720_setting(radix=radix,bit_width=bit_width,rotate=rotate,crop_fisrt=crop_fisrt,round_w_to_16=round_w_to_16,NUM_BANK_LINE=NUM_BANK_LINE,BANK_ENTRY_CNT=BANK_ENTRY_CNT,MAX_IMG_PREPROC_ROW_NUM=MAX_IMG_PREPROC_ROW_NUM,MAX_IMG_PREPROC_COL_NUM=MAX_IMG_PREPROC_COL_NUM)
    image_data, _ = flow_720.run_whole_process(image)

    return image_data

def bit_match(data1, data2):
    """
    bit_match function

    check data1 is equal to data2 or not.

    Args:
        data1: [np.array / str], can be array or txt/bin file
        data2: [np.array / str], can be array or txt/bin file

    Returns:
        out1: [bool], is match or not
        out2: [np.array], if not match, save the position for mismatched data

    Examples:
        >>> result, mismatched = kneron_preprocessing.API.bit_match(data1,data2)
    """
    if isinstance(data1, str):
        if os.path.splitext(data1)[1] == '.bin':
            data1 = np.fromfile(data1, dtype='uint8')
        elif os.path.splitext(data1)[1] == '.txt':
            data1 = np.loadtxt(data1)
    
    assert isinstance(data1, np.ndarray)

    if isinstance(data2, str):
        if os.path.splitext(data2)[1] == '.bin':
            data2 = np.fromfile(data2, dtype='uint8')
        elif os.path.splitext(data2)[1] == '.txt':
            data2 = np.loadtxt(data2)

    assert isinstance(data2, np.ndarray)


    data1 = data1.reshape((-1,1))
    data2 = data2.reshape((-1,1))

    if not(len(data1) == len(data2)):
        print('error len')
        return False, np.zeros((1))
    else: 
        ans = data2 - data1    
        if len(np.where(ans>0)[0]) > 0:
            print('error',np.where(ans>0)[0])
            return False, np.where(ans>0)[0]
        else:
            print('pass')
            return True, np.zeros((1))

def cpr_to_crp(x_start, x_end, y_start, y_end, pad_l, pad_r, pad_t, pad_b, rx_start, rx_end, ry_start, ry_end):
    """
    calculate the parameters of crop->pad->resize flow  to HW crop->resize->padding flow

    Args:

    Returns:

    Examples:

    """
    pad_l = round(pad_l * (rx_end-rx_start) / (x_end - x_start + pad_l + pad_r))
    pad_r = round(pad_r * (rx_end-rx_start) / (x_end - x_start + pad_l + pad_r)) 
    pad_t = round(pad_t * (ry_end-ry_start) / (y_end - y_start + pad_t + pad_b))
    pad_b = round(pad_b * (ry_end-ry_start) / (y_end - y_start + pad_t + pad_b))

    rx_start +=pad_l
    rx_end -=pad_r
    ry_start +=pad_t
    ry_end -=pad_b

    return x_start, x_end, y_start, y_end, pad_l, pad_r, pad_t, pad_b, rx_start, rx_end, ry_start, ry_end