import numpy as np
from PIL import Image
from .utils import str2bool, str2int, str2float
from .Runner_base import Runner_base, Param_base

class General(Param_base):
    type = ''
    pad_val = ''
    padded_w = ''
    padded_h = ''
    pad_l = ''
    pad_r = ''
    pad_t = ''
    pad_b = ''
    padding_ch = 3
    padding_ch_type = 'RGB'
    def update(self, **dic):
        self.type = dic['type']
        self.pad_val = dic['pad_val']
        self.padded_w = str2int(dic['padded_w'])
        self.padded_h = str2int(dic['padded_h'])
        self.pad_l = str2int(dic['pad_l'])
        self.pad_r = str2int(dic['pad_r'])
        self.pad_t = str2int(dic['pad_t'])
        self.pad_b = str2int(dic['pad_b'])

    def __str__(self):
        str_out = [
            ', type:',str(self.type),
            ', pad_val:',str(self.pad_val),
            ', pad_l:',str(self.pad_l),
            ', pad_r:',str(self.pad_r),
            ', pad_r:',str(self.pad_t),
            ', pad_b:',str(self.pad_b),
            ', padding_ch:',str(self.padding_ch)]
        return(' '.join(str_out))

class Hw(Param_base):
    radix = 8
    normalize_type = 'floating'
    def update(self, **dic):
        self.radix = dic['radix']
        self.normalize_type = dic['normalize_type']

    def __str__(self):
        str_out = [
            ', radix:', str(self.radix),
            ', normalize_type:',str(self.normalize_type)]
        return(' '.join(str_out))


class runner(Runner_base):
    ## overwrite the class in Runner_base
    general = General()
    hw = Hw()

    def __str__(self):
        return('<Padding>')

    def update(self, **kwargs):
        super().update(**kwargs)

        ## update pad type & pad length
        if (self.general.pad_l != 0) or (self.general.pad_r != 0) or (self.general.pad_t != 0) or (self.general.pad_b != 0):
            self.general.type = 'specific'
            assert(self.general.pad_l >= 0)
            assert(self.general.pad_r >= 0)
            assert(self.general.pad_t >= 0)
            assert(self.general.pad_b >= 0)
        elif(self.general.type != 'specific'):
            if self.general.padded_w == 0 or self.general.padded_h == 0:
                self.general.padded_w = self.common.model_size[0]
                self.general.padded_h = self.common.model_size[1]
            assert(self.general.padded_w > 0)
            assert(self.general.padded_h > 0)
            assert(self.general.type.lower() in ['CENTER', 'Center', 'center', 'CORNER', 'Corner', 'corner'])
        else:
            assert(self.general.type == 'specific')
            
        ## decide pad_val & padding ch
        # if numerical_type is floating
        if (self.common.numerical_type == 'floating'):
            if self.general.pad_val != 'edge':
                self.general.pad_val = str2float(self.general.pad_val)
            self.general.padding_ch = 3
            self.general.padding_ch_type = 'RGB'
        # if numerical_type is 520 or 720
        else: 
            if self.general.pad_val == '':
                if self.hw.normalize_type.lower() in ['TF', 'Tf', 'tf']:
                    self.general.pad_val = np.uint8(-128 >> (7 - self.hw.radix))
                elif self.hw.normalize_type.lower() in ['YOLO', 'Yolo', 'yolo']:
                    self.general.pad_val = np.uint8(0 >> (8 - self.hw.radix))
                elif self.hw.normalize_type.lower() in ['KNERON', 'Kneron', 'kneron']:
                    self.general.pad_val = np.uint8(-128 >> (8 - self.hw.radix))
                else:
                    self.general.pad_val = np.uint8(0 >> (8 - self.hw.radix))
            else:
                self.general.pad_val = str2int(self.general.pad_val)
            self.general.padding_ch = 4
            self.general.padding_ch_type = 'RGBA'

    def run(self, image_data):
        # init
        shape = image_data.shape
        w = shape[1]
        h = shape[0]
        if len(shape) < 3:
            self.general.padding_ch = 1
            self.general.padding_ch_type = 'L'
        else:
            if shape[2] == 3 and self.general.padding_ch == 4:
                image_data = np.concatenate((image_data, np.zeros((h, w, 1), dtype=np.uint8) ), axis=2)
                
        ## padding
        if self.general.type.lower() in ['CENTER',  'Center',  'center']:
            img_pad = self._padding_center(image_data, w, h)
        elif self.general.type.lower() in ['CORNER',  'Corner',  'corner']:
            img_pad = self._padding_corner(image_data, w, h)
        else:
            img_pad = self._padding_sp(image_data, w, h)

        # print info
        if str2bool(self.common.print_info):
            self.print_info()

        # output
        info = {}
        return img_pad, info

    ## protect fun
    def _padding_center(self, img, ori_w, ori_h):
        # img_pad = Image.new(self.general.padding_ch_type, (self.general.padded_w, self.general.padded_h), int(self.general.pad_val[0]))
        # img = Image.fromarray(img)
        # img_pad.paste(img, ((self.general.padded_w-ori_w)//2, (self.general.padded_h-ori_h)//2))
        # return img_pad
        padH = self.general.padded_h - ori_h
        padW = self.general.padded_w - ori_w
        self.general.pad_t = padH // 2
        self.general.pad_b = (padH // 2) + (padH % 2)
        self.general.pad_l = padW // 2
        self.general.pad_r = (padW // 2) + (padW % 2)
        if self.general.pad_l < 0 or self.general.pad_r <0 or self.general.pad_t <0 or self.general.pad_b<0:
            return img
        img_pad = self._padding_sp(img,ori_w,ori_h)
        return img_pad

    def _padding_corner(self, img, ori_w, ori_h):
        # img_pad = Image.new(self.general.padding_ch_type, (self.general.padded_w, self.general.padded_h), self.general.pad_val)
        # img_pad.paste(img, (0, 0))
        self.general.pad_l = 0
        self.general.pad_r = self.general.padded_w - ori_w
        self.general.pad_t = 0
        self.general.pad_b = self.general.padded_h - ori_h
        if self.general.pad_l < 0 or self.general.pad_r <0 or self.general.pad_t <0 or self.general.pad_b<0:
            return img
        img_pad = self._padding_sp(img,ori_w,ori_h)
        return img_pad

    def _padding_sp(self, img, ori_w, ori_h):
        # block_t = np.zeros((self.general.pad_t, self.general.pad_l + self.general.pad_r + ori_w, self.general.padding_ch), dtype=np.float)
        # block_l = np.zeros((ori_h, self.general.pad_l, self.general.padding_ch), dtype=np.float)
        # block_r = np.zeros((ori_h, self.general.pad_r, self.general.padding_ch), dtype=np.float)
        # block_b = np.zeros((self.general.pad_b, self.general.pad_l + self.general.pad_r + ori_w, self.general.padding_ch), dtype=np.float)
        # for i in range(self.general.padding_ch):
        #     block_t[:, :, i] = np.ones(block_t[:, :, i].shape, dtype=np.float) * self.general.pad_val
        #     block_l[:, :, i] = np.ones(block_l[:, :, i].shape, dtype=np.float) * self.general.pad_val
        #     block_r[:, :, i] = np.ones(block_r[:, :, i].shape, dtype=np.float) * self.general.pad_val
        #     block_b[:, :, i] = np.ones(block_b[:, :, i].shape, dtype=np.float) * self.general.pad_val
        # padded_image_hor = np.concatenate((block_l, img, block_r), axis=1)
        # padded_image = np.concatenate((block_t, padded_image_hor, block_b), axis=0)
        # return padded_image
        if self.general.padding_ch == 1:
            pad_range = ( (self.general.pad_t, self.general.pad_b),(self.general.pad_l, self.general.pad_r) )
        else:
            pad_range = ((self.general.pad_t, self.general.pad_b),(self.general.pad_l, self.general.pad_r),(0,0))

        if isinstance(self.general.pad_val, str):
            if self.general.pad_val == 'edge':
                padded_image = np.pad(img, pad_range, mode="edge")
            else:
                padded_image = np.pad(img, pad_range, mode="constant",constant_values=0)
        else:
            padded_image = np.pad(img, pad_range, mode="constant",constant_values=self.general.pad_val)
        
        return padded_image

