# coding: utf-8
import cv2
import numpy as np
import math
from . import kneron_preprocessing
kneron_preprocessing.API.set_default_as_520()

img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']
def make_divisible(x, divisor):
    # Returns x evenly divisble by divisor
    return math.ceil(x / divisor) * divisor

def check_img_size(img_size, s=32):
    # Verify img_size is a multiple of stride s
    new_size = make_divisible(img_size, int(s))  # ceil gs-multiple
    if new_size != img_size:
        print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
    return new_size

def letterbox(img, new_shape=(640, 640), color=(0, 0, 0), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r)) # width, height 
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if shape[::-1] != new_unpad:  # resize
        img = kneron_preprocessing.API.resize(img,size=new_unpad, keep_ratio = False)

    top, bottom = int(0), int(round(dh + 0.1))
    left, right = int(0), int(round(dw + 0.1))    

    img = kneron_preprocessing.API.pad(img, left, right, top, bottom, 0)

    return img, ratio, (dw, dh)

def resize(img, new_shape=(640, 640), color=(0, 0, 0), auto=True, scaleFill=False, scaleup=True):

    ratio = 1.0, 1.0
    dw, dh = 0, 0
    img = kneron_preprocessing.API.resize(img, size=new_shape, keep_ratio=False, type='bilinear')
    return img, ratio, (dw, dh)

def LoadImages(path,img_size,keep_ratio):  #_rgb # for inference
    if isinstance(path, str):
        img0 = cv2.imread(path)  # BGR       
    else:
        img0 = path  # BGR

    # Padded resize
    if keep_ratio:
        img = letterbox(img0, new_shape=img_size)[0]
    else:
        img = resize(img0, new_shape=img_size)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img)
    return img, img0

def preprocess(image_path, imgsz_h, imgsz_w, keep_ratio=True) :
    model_stride_max = 32
    imgsz_h = check_img_size(imgsz_h, s=model_stride_max)  # check img_size
    imgsz_w = check_img_size(imgsz_w, s=model_stride_max)  # check img_size
    img, im0 = LoadImages(image_path, img_size=(imgsz_h,imgsz_w), keep_ratio=keep_ratio)
    img = kneron_preprocessing.API.norm(img)
    #print('img',img.shape)

    #add batch (1, c, h, w)
    if img.ndim == 3:
        img = img.reshape(1, *img.shape)

    return img, im0

