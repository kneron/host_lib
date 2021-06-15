import numpy as np
from PIL import Image
from .utils import str2int, str2float, str2bool, pad_square_to_4
from .utils_520 import round_up_n
from .Runner_base import Runner_base, Param_base

class General(Param_base):
    type = 'center'
    align_w_to_4 = False
    pad_square_to_4 = False
    rounding_type = 0
    crop_w = 0
    crop_h = 0
    start_x = 0.
    start_y = 0.
    end_x = 0.
    end_y = 0.
    def update(self, **dic):
        self.type = dic['type']
        self.align_w_to_4 = str2bool(dic['align_w_to_4'])
        self.rounding_type = str2int(dic['rounding_type'])
        self.crop_w = str2int(dic['crop_w'])
        self.crop_h = str2int(dic['crop_h'])
        self.start_x = str2float(dic['start_x'])
        self.start_y = str2float(dic['start_y'])
        self.end_x = str2float(dic['end_x'])
        self.end_y = str2float(dic['end_y'])

    def __str__(self):
        str_out = [
            ', type:',str(self.type),
            ', align_w_to_4:',str(self.align_w_to_4),
            ', pad_square_to_4:',str(self.pad_square_to_4),
            ', crop_w:',str(self.crop_w),
            ', crop_h:',str(self.crop_h),
            ', start_x:',str(self.start_x),
            ', start_y:',str(self.start_y),
            ', end_x:',str(self.end_x),
            ', end_y:',str(self.end_y)]
        return(' '.join(str_out))
       
class runner(Runner_base):
    ## overwrite the class in Runner_base
    general = General()

    def __str__(self):
        return('<Crop>')

    def update(self, **kwargs):
        ##
        super().update(**kwargs)

        ##
        if (self.general.start_x != self.general.end_x) and (self.general.start_y != self.general.end_y):
            self.general.type = 'specific'
        elif(self.general.type != 'specific'):
            if self.general.crop_w == 0 or self.general.crop_h == 0:
                self.general.crop_w = self.common.model_size[0]
                self.general.crop_h = self.common.model_size[1]
            assert(self.general.crop_w > 0)
            assert(self.general.crop_h > 0)
            assert(self.general.type.lower() in ['CENTER', 'Center', 'center', 'CORNER', 'Corner', 'corner'])
        else:
            assert(self.general.type == 'specific')

    def run(self, image_data):
        ## init
        img = Image.fromarray(image_data)
        w, h = img.size

        ## get range
        if self.general.type.lower() in ['CENTER', 'Center', 'center']:
            x1, y1, x2, y2 = self._calcuate_xy_center(w, h)
        elif self.general.type.lower() in ['CORNER', 'Corner', 'corner']:
            x1, y1, x2, y2 = self._calcuate_xy_corner(w, h)
        else:
            x1 = self.general.start_x
            y1 = self.general.start_y
            x2 = self.general.end_x
            y2 = self.general.end_y
            assert( ((x1 != x2) and (y1 != y2)) )

        ## rounding
        if self.general.rounding_type == 0:
            x1 = int(np.floor(x1))
            y1 = int(np.floor(y1))
            x2 = int(np.ceil(x2))
            y2 = int(np.ceil(y2))
        else:
            x1 = int(round(x1))
            y1 = int(round(y1))
            x2 = int(round(x2))
            y2 = int(round(y2))

        if self.general.align_w_to_4:
            # x1 = (x1+1) &(~3)  #//+2
            # x2 = (x2+2) &(~3)  #//+1
            x1 = (x1+3) &(~3)  #//+2
            left = w - x2
            left = (left+3) &(~3)
            x2 = w - left

        ## pad_square_to_4
        if str2bool(self.general.pad_square_to_4):
            x1,x2,y1,y2 = pad_square_to_4(x1,x2,y1,y2)

        # do crop
        box = (x1,y1,x2,y2)
        img = img.crop(box)

        # print info
        if str2bool(self.common.print_info):
            self.general.start_x = x1
            self.general.start_y = y1
            self.general.end_x = x2
            self.general.end_y = y2
            self.general.crop_w = x2 - x1
            self.general.crop_h = y2 - y1
            self.print_info()

        # output
        image_data = np.array(img)
        info = {}
        info['box'] = box

        return image_data, info


    ## protect fun
    def _calcuate_xy_center(self, w, h):
        x1 = w/2 - self.general.crop_w / 2
        y1 = h/2 - self.general.crop_h / 2
        x2 = w/2 + self.general.crop_w / 2
        y2 = h/2 + self.general.crop_h / 2
        return x1, y1, x2, y2

    def _calcuate_xy_corner(self, _1, _2):
        x1 = 0
        y1 = 0
        x2 = self.general.crop_w
        y2 = self.general.crop_h
        return x1, y1, x2, y2

    def do_crop(self, image_data, startW, startH, endW, endH):
        return image_data[startH:endH, startW:endW, :]
