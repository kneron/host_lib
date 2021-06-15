import numpy as np
import cv2
from PIL import Image
from .utils import str2bool, str2int
from ctypes import c_float
from .Runner_base import Runner_base, Param_base

class General(Param_base):
    type = 'bilinear'
    keep_ratio = True
    zoom = True
    calculate_ratio_using_CSim = True
    resize_w = 0
    resize_h = 0
    resized_w = 0
    resized_h = 0
    def update(self, **dic):
        self.type = dic['type']
        self.keep_ratio = str2bool(dic['keep_ratio'])
        self.zoom = str2bool(dic['zoom'])
        self.calculate_ratio_using_CSim = str2bool(dic['calculate_ratio_using_CSim'])
        self.resize_w = str2int(dic['resize_w'])
        self.resize_h = str2int(dic['resize_h'])

    def __str__(self):
        str_out = [
            ', type:',str(self.type),
            ', keep_ratio:',str(self.keep_ratio),
            ', zoom:',str(self.zoom),
            ', calculate_ratio_using_CSim:',str(self.calculate_ratio_using_CSim),
            ', resize_w:',str(self.resize_w),
            ', resize_h:',str(self.resize_h),
            ', resized_w:',str(self.resized_w),
            ', resized_h:',str(self.resized_h)]
        return(' '.join(str_out))

class Hw(Param_base):
    resize_bit = 12
    def update(self, **dic):
        pass

    def __str__(self):
        str_out = [
            ', resize_bit:',str(self.resize_bit)]
        return(' '.join(str_out))

class runner(Runner_base):
    ## overwrite the class in Runner_base
    general = General()
    hw = Hw()

    def __str__(self):
        return('<Resize>')

    def update(self, **kwargs):
        super().update(**kwargs)
        
        ## if resize size has not been assigned, then it will take model size as resize size
        if self.general.resize_w == 0 or self.general.resize_h == 0:
            self.general.resize_w = self.common.model_size[0]
            self.general.resize_h = self.common.model_size[1]
        assert(self.general.resize_w > 0)
        assert(self.general.resize_h > 0)

        ##
        if self.common.numerical_type == '520':
            self.general.type = 'fixed_520'
        elif self.common.numerical_type == '720':
            self.general.type = 'fixed_720'
        assert(self.general.type.lower() in ['BILINEAR',  'Bilinear',  'bilinear', 'BICUBIC',  'Bicubic',  'bicubic', 'FIXED',  'Fixed', 'fixed', 'FIXED_520',  'Fixed_520',  'fixed_520', 'FIXED_720', 'Fixed_720', 'fixed_720','CV', 'cv', 'opencv', 'OpenCV', 'CV2', 'cv2'])


    def run(self, image_data):
        ## init
        ori_w = image_data.shape[1]
        ori_h = image_data.shape[0]
        info = {}

        ##
        if self.general.keep_ratio:
            self.general.resized_w, self.general.resized_h = self.calcuate_scale_keep_ratio(self.general.resize_w,self.general.resize_h, ori_w, ori_h, self.general.calculate_ratio_using_CSim)
        else:
            self.general.resized_w = int(self.general.resize_w)
            self.general.resized_h = int(self.general.resize_h)
        assert(self.general.resized_w > 0)
        assert(self.general.resized_h > 0)

        ##
        if (self.general.resized_w > ori_w) or (self.general.resized_h > ori_h):
            if not self.general.zoom: 
                info['size'] = (ori_w,ori_h)
                if str2bool(self.common.print_info):
                    print('no resize')
                    self.print_info()
                return image_data, info

        ## resize
        if self.general.type.lower() in ['BILINEAR',  'Bilinear',  'bilinear']:
            image_data = self.do_resize_bilinear(image_data, self.general.resized_w, self.general.resized_h)
        elif self.general.type.lower() in ['BICUBIC',  'Bicubic',  'bicubic']:
            image_data = self.do_resize_bicubic(image_data, self.general.resized_w, self.general.resized_h)
        elif self.general.type.lower() in ['CV',  'cv',  'opencv', 'OpenCV',  'CV2',  'cv2']:
            image_data = self.do_resize_cv2(image_data, self.general.resized_w, self.general.resized_h)
        elif self.general.type.lower() in ['FIXED',  'Fixed',  'fixed', 'FIXED_520',  'Fixed_520',  'fixed_520', 'FIXED_720', 'Fixed_720', 'fixed_720']:
            image_data = self.do_resize_fixed(image_data, self.general.resized_w, self.general.resized_h, self.hw.resize_bit, self.general.type)

       
        # output
        info['size'] = (self.general.resized_w, self.general.resized_h)

        # print info
        if str2bool(self.common.print_info):
            self.print_info()

        return image_data, info

    def calcuate_scale_keep_ratio(self, tar_w, tar_h, ori_w, ori_h, calculate_ratio_using_CSim):
        if not calculate_ratio_using_CSim:
            scale_w = tar_w * 1.0 / ori_w*1.0
            scale_h = tar_h * 1.0 / ori_h*1.0
            scale = scale_w if scale_w < scale_h else scale_h
            new_w = int(round(ori_w * scale))
            new_h = int(round(ori_h * scale))
            return new_w, new_h
        
        ## calculate_ratio_using_CSim
        scale_w = c_float(tar_w * 1.0 / (ori_w * 1.0)).value
        scale_h = c_float(tar_h * 1.0 / (ori_h * 1.0)).value
        scale_ratio = 0.0
        scale_target_w = 0
        scale_target_h = 0
        padH = 0
        padW = 0

        bScaleW = True if scale_w < scale_h else False
        if bScaleW:
            scale_ratio = scale_w
            scale_target_w = int(c_float(scale_ratio * ori_w + 0.5).value)
            scale_target_h = int(c_float(scale_ratio * ori_h + 0.5).value)
            assert (abs(scale_target_w - tar_w) <= 1), "Error: scale down width cannot meet expectation\n"
            padH = tar_h - scale_target_h
            padW = 0
            assert (padH >= 0), "Error: padH shouldn't be less than zero\n"
        else:
            scale_ratio = scale_h 
            scale_target_w = int(c_float(scale_ratio * ori_w + 0.5).value)
            scale_target_h = int(c_float(scale_ratio * ori_h + 0.5).value)
            assert (abs(scale_target_h - tar_h) <= 1), "Error: scale down height cannot meet expectation\n"
            padW = tar_w - scale_target_w
            padH = 0
            assert (padW >= 0), "Error: padW shouldn't be less than zero\n"
        new_w = tar_w - padW
        new_h = tar_h - padH
        return new_w, new_h
    
    def do_resize_bilinear(self, image_data, resized_w, resized_h):
        img = Image.fromarray(image_data)
        img = img.resize((resized_w, resized_h), Image.BILINEAR)
        image_data = np.array(img).astype('uint8')
        return image_data        

    def do_resize_bicubic(self, image_data, resized_w, resized_h):
        img = Image.fromarray(image_data)
        img = img.resize((resized_w, resized_h), Image.BICUBIC)
        image_data = np.array(img).astype('uint8')
        return image_data

    def do_resize_cv2(self, image_data, resized_w, resized_h):
        image_data = cv2.resize(image_data, (resized_w, resized_h))
        image_data = np.array(image_data)
        # image_data = np.array(image_data).astype('uint8')
        return image_data

    def do_resize_fixed(self, image_data, resized_w, resized_h, resize_bit, type):
        if len(image_data.shape) < 3:
            m, n = image_data.shape
            tmp = np.zeros((m,n,3), dtype=np.uint8)
            tmp[:,:,0] = image_data
            image_data = tmp
            c = 3
            gray = True
        else:
            m, n, c = image_data.shape
            gray = False

        resolution = 1 << resize_bit

        # Width
        ratio = int(((n - 1) << resize_bit) / (resized_w - 1))
        ratio_cnt = 0
        src_x = 0
        resized_image_w = np.zeros((m, resized_w, c), dtype=np.uint8)
        
        for dst_x in range(resized_w):
            while ratio_cnt > resolution:
                ratio_cnt = ratio_cnt - resolution
                src_x = src_x + 1
            mul1 = np.ones((m, c)) * (resolution - ratio_cnt)
            mul2 = np.ones((m, c)) * ratio_cnt
            resized_image_w[:, dst_x, :] = np.multiply(np.multiply(
                image_data[:, src_x, :], mul1) + np.multiply(image_data[:, src_x + 1, :], mul2), 1/resolution)
            ratio_cnt = ratio_cnt + ratio

        # Height
        ratio = int(((m - 1) << resize_bit) / (resized_h - 1))
        ## NPU HW special case 2 , only on 520
        if type.lower() in ['FIXED_520',  'Fixed_520',  'fixed_520']:
            if (((ratio * (resized_h - 1)) % 4096 == 0) and ratio != 4096):
                ratio -= 1

        ratio_cnt = 0
        src_x = 0
        resized_image = np.zeros(
            (resized_h, resized_w, c), dtype=np.uint8)
        for dst_x in range(resized_h):
            while ratio_cnt > resolution:
                ratio_cnt = ratio_cnt - resolution
                src_x = src_x + 1
                       
            mul1 = np.ones((resized_w, c)) * (resolution - ratio_cnt)
            mul2 = np.ones((resized_w, c)) * ratio_cnt
            
            ## NPU HW special case 1 , both on 520 / 720
            if (((dst_x > 0) and ratio_cnt == resolution) and (ratio != resolution)):
                if type.lower() in ['FIXED_520',  'Fixed_520',  'fixed_520','FIXED_720',  'Fixed_720',  'fixed_720' ]:
                    resized_image[dst_x, :, :] = np.multiply(np.multiply(
                        resized_image_w[src_x+1, :, :], mul1) + np.multiply(resized_image_w[src_x + 2, :, :], mul2), 1/resolution)
            else:
                resized_image[dst_x, :, :] = np.multiply(np.multiply(
                    resized_image_w[src_x, :, :], mul1) + np.multiply(resized_image_w[src_x + 1, :, :], mul2), 1/resolution)

            ratio_cnt = ratio_cnt + ratio

        if gray:
            resized_image = resized_image[:,:,0]

        return resized_image
