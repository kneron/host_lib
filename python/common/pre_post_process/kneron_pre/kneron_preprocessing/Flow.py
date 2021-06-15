import numpy as np
from PIL import Image
import json
import math
import sys
from .funcs import *
from .funcs.utils import str2bool, bin_loader, hex_loader, str_fill, clip_ary
from .funcs.utils_520 import round_up_16, round_up_n, cal_img_row_offset, get_pad_num, get_byte_per_pixel
from .funcs.utils_720 import twos_complement_pix, clip_pix
from ctypes import c_float


class Flow(object):
    # class function
    def __init__(self, config_path = ''):
        '''
        @brief:
        Class name: Flow
        Constructor with config_path

        @param:
        config_path[str]: json file path or empty, init this class with json file. If empty, will use default setting.
        '''
        # init config
        self.__init_config()

        # update config with joson file
        try:
            with open(config_path, encoding='utf-8') as f:
                self.config = json.load(f)
        except IOError:
            pass

         # print info
        if str2bool(self.config['print_info']):
            print("pre-processing type:", self.config['type_name'],", model_size:",self.config['model_size'],", numerical_type",self.config['numerical_type'])
        
        # init funcs
        self.error_state = 0
        self.subclass = {}
        self.subclass['color'] = ColorConversion.runner()
        self.subclass['resize'] = Resize.runner()
        self.subclass['crop'] = Crop.runner()
        self.subclass['padding'] = Padding.runner()
        self.subclass['normalize'] = Normalize.runner()

        self.funcs = {}
        self.funcs['crop'] = self.run_crop
        self.funcs['color'] = self.run_color_conversion
        self.funcs['resize'] = self.run_resize
        self.funcs['normalize'] = self.run_normalize
        self.funcs['padding'] = self.run_padding

        return

    def __init_config(self):
        '''
        private function
        '''
        self.config = {
            "_comment": "PreProcessing",
            "type_name": "default",
            "numerical_type": "floating",
            "print_info":"no",
            "model_size": [
                56,
                56
            ],
            "raw_img":{
                "is_raw_img": "no",
                "raw_img_type": "bin",
                "raw_img_fmt": "rgb565",
                "img_in_width": 640,
                "img_in_height": 480
            },
            "output_setting":{
                "is_dump": "no",
                "dump_format":"bin",
                "output_file":"default.bin",
                "image_format":"RGB888"
            },
            "520_setting":{
                "radix": 8,
                "bit_width": 8,
                "rotate": 0,
                "crop_fisrt": "no",
                "NUM_BANK_LINE": 32,
                "BANK_ENTRY_CNT": 512,
                "MAX_IMG_PREPROC_ROW_NUM": 511,
                "MAX_IMG_PREPROC_COL_NUM": 256,
                "round_w_to_16": "no"
            },
            "720_setting":{
                "radix": 8,
                "shift":0,
                "sub":0,
                "bit_width": 8,
                "rotate": 0,
                "crop_fisrt": "no",
                "matrix_c00": 1,
                "matrix_c01": 0,
                "matrix_c02": 0,
                "matrix_c10": 0,
                "matrix_c11": 1,
                "matrix_c12": 0,
                "matrix_c20": 0,
                "matrix_c21": 0,
                "matrix_c22": 1,
                "vector_b00": 0,
                "vector_b01": 0,
                "vector_b02": 0
            },
            "floating_setting":{
                "job_list":[     
                    "color",
                    "crop",
                    "resize",
                    "padding",
                    "normalize",
                    ]
            },
            "function_setting": {
                "color": {
                    "out_format": "rgb888",
                    "options": {
                        "simulation": "no",
                        "simulation_format": ""
                    }
                },
                "crop": {
                    "type": "corner",
                    "align_w_to_4":"no",
                    "pad_square_to_4":"no",
                    "rounding_type":0,
                    "crop_w": "",
                    "crop_h": "",
                    "start_x": "",
                    "start_y": "",
                    "end_x": "",
                    "end_y": ""
                },
                "resize": {
                    "type": "fixed",
                    "keep_ratio": "yes",
                    "calculate_ratio_using_CSim": "yes",
                    "zoom": "yes",
                    "resize_w": "",
                    "resize_h": "",
                },
                "padding": {
                    "type": "corner",
                    "pad_val": "",
                    "padded_w": "",
                    "padded_h": "",
                    "pad_l": "",
                    "pad_r": "",
                    "pad_t": "",
                    "pad_b": ""
                },
                "normalize": {
                    "type": "kneron",
                    "scale": "",
                    "bias": "",
                    "mean": "",
                    "std": ""
                }
            }
        }
        return
    
    def __update_color(self):
        '''
        private function
        '''
        #
        dic = self.config['function_setting']['color']
        dic['model_size'] = self.config['model_size']
        dic['print_info'] = self.config['print_info']
        self.subclass['color'].update(**dic)

        return

    def __update_crop(self):
        '''
        private function
        '''
        dic = {}
        # common
        dic['common'] = {}
        dic['common']['print_info'] = self.config['print_info']
        dic['common']['model_size'] = self.config['model_size']
        dic['common']['numerical_type'] = self.config['numerical_type']

        # general
        dic['general'] = {}
        dic['general']['type'] = self.config['function_setting']['crop']['type']
        dic['general']['align_w_to_4'] = self.config['function_setting']['crop']['align_w_to_4']
        dic['general']['pad_square_to_4'] = self.config['function_setting']['crop']['pad_square_to_4']
        dic['general']['rounding_type'] = self.config['function_setting']['crop']['rounding_type']
        dic['general']['crop_w'] = self.config['function_setting']['crop']['crop_w']
        dic['general']['crop_h'] = self.config['function_setting']['crop']['crop_h']
        dic['general']['start_x'] = self.config['function_setting']['crop']['start_x']
        dic['general']['start_y'] = self.config['function_setting']['crop']['start_y']
        dic['general']['end_x'] = self.config['function_setting']['crop']['end_x']
        dic['general']['end_y'] = self.config['function_setting']['crop']['end_y']
       
        # floating
        dic['floating'] = {}

        # hw
        dic['hw'] = {}
        
        
        self.subclass['crop'].update(**dic)
        return
    
    def __update_resize(self):
        '''
        private function
        '''
        dic = {}
        # common
        dic['common'] = {}
        dic['common']['print_info'] = self.config['print_info']
        dic['common']['model_size'] = self.config['model_size']
        dic['common']['numerical_type'] = self.config['numerical_type']

        # general
        dic['general'] = {}
        dic['general']['type'] = self.config['function_setting']['resize']['type']
        dic['general']['keep_ratio'] = self.config['function_setting']['resize']['keep_ratio']
        dic['general']['zoom'] = self.config['function_setting']['resize']['zoom']
        dic['general']['calculate_ratio_using_CSim'] = self.config['function_setting']['resize']['calculate_ratio_using_CSim']
        dic['general']['resize_w'] = self.config['function_setting']['resize']['resize_w']
        dic['general']['resize_h'] = self.config['function_setting']['resize']['resize_h']
       
        # floating
        dic['floating'] = {}

        # hw
        dic['hw'] = {}

        self.subclass['resize'].update(**dic)
        return

    def __update_normalize(self):
        '''
        private function
        '''
        dic = {}
        # general
        dic['general'] = {}
        dic['general']['print_info'] = self.config['print_info']
        dic['general']['model_size'] = self.config['model_size']
        dic['general']['numerical_type'] = self.config['numerical_type']
        dic['general']['type'] = self.config['function_setting']['normalize']['type']

        # floating
        dic['floating'] = {}
        dic['floating']['scale'] = self.config['function_setting']['normalize']['scale']
        dic['floating']['bias'] = self.config['function_setting']['normalize']['bias']
        dic['floating']['mean'] = self.config['function_setting']['normalize']['mean']
        dic['floating']['std'] = self.config['function_setting']['normalize']['std']

        # hw
        dic['hw'] = {}
        if self.config['numerical_type'] == '520':
            dic['hw']['radix'] = self.config['520_setting']['radix']
        if self.config['numerical_type'] == '720':
            dic['hw']['radix'] = self.config['720_setting']['radix']

        self.subclass['normalize'].update(**dic)
        return

    def __update_padding(self):
        '''
        private function
        '''
        dic = {}
        # common
        dic['common'] = {}
        dic['common']['print_info'] = self.config['print_info']
        dic['common']['model_size'] = self.config['model_size']
        dic['common']['numerical_type'] = self.config['numerical_type']

        # general
        dic['general'] = {}
        dic['general']['type'] = self.config['function_setting']['padding']['type']
        dic['general']['pad_val'] = self.config['function_setting']['padding']['pad_val']
        dic['general']['padded_w'] = self.config['function_setting']['padding']['padded_w']
        dic['general']['padded_h'] = self.config['function_setting']['padding']['padded_h']
        dic['general']['pad_l'] = self.config['function_setting']['padding']['pad_l']
        dic['general']['pad_r'] = self.config['function_setting']['padding']['pad_r']
        dic['general']['pad_t'] = self.config['function_setting']['padding']['pad_t']
        dic['general']['pad_b'] = self.config['function_setting']['padding']['pad_b']

        # floating
        dic['floating'] = {}

        # hw
        dic['hw'] = {}
        if self.config['numerical_type'] == '520':
            dic['hw']['radix'] = self.config['520_setting']['radix']
            dic['hw']['normalize_type'] = self.config['function_setting']['normalize']['type']
        elif self.config['numerical_type'] == '720':
            dic['hw']['radix'] = self.config['720_setting']['radix']
            dic['hw']['normalize_type'] = self.config['function_setting']['normalize']['type']

        self.subclass['padding'].update(**dic)
        return

    def set_numerical_type(self, type = ''):
        '''
        set_numerical_type
        
        set the preprocess type, now support floating, 520 and 720

        Args:
            type: [str], "520" / "720" / "floating"
        '''
        if not (type.lower() in ['520', '720', 'floating']):
            type = 'floating'
        self.config['numerical_type'] = type
        return    

    def set_print_info(self, print_info = ''):
        '''
        turn print infomation on or off.

        Args:
            print_info: [str], "yes" / "no"
        '''
        self.config['print_info'] = print_info
        return

    def set_model_size(self, w, h):
        '''
        set_model_size, set out image size, or npu size

        Args:
            w: [int]
            h: [int]
        '''
        if w <= 0 or h <= 0:
            return
        self.config['model_size'][0] = w
        self.config['model_size'][1] = h
 
        return

    def set_raw_img(self, is_raw_img='', raw_img_type = '', raw_img_fmt='', img_in_width='',img_in_height=''):
        '''
        set if input is raw file

        now support for rgb888,rgb565,nir,yuv and ycbcr

        Args:
            is_raw_img: [str], "yes" / "no", is raw file or not 
            raw_img_type: [str], "bin" / "hex", set the raw file format, now support bin and hex file.
            raw_img_fmt: [str], "rgb888" / "rgb565" / "nir" / "ycbcr422" / "ycbcr444" / "yuv422" / "yuv444", set the raw image format.
            img_in_width: [int]
            img_in_height: [int]
        '''
        if not(is_raw_img==''):
            self.config['raw_img']['is_raw_img'] = is_raw_img
        if not(raw_img_type==''):
            self.config['raw_img']['raw_img_type'] = raw_img_type             
        if not(raw_img_fmt==''):
            self.config['raw_img']['raw_img_fmt'] = raw_img_fmt        
        if not(img_in_width==''):
            self.config['raw_img']['img_in_width'] = img_in_width        
        if not(img_in_height==''):
            self.config['raw_img']['img_in_height'] = img_in_height 
        return

    def set_output_setting(self, is_dump='', dump_format='',image_format='', output_file=''):
        '''
        set_output_setting, dump output or not, dump format can be bin , hex or txt

        Args:
            is_dump: [str], "yes" / "no", open dump function or not
            dump_format: [str], "bin" / "txt" / "hex", set dump file format.
            image_format: [str], RGB888 / RGBA8888 / RGB565 / NIR / YUV444 / YCbCr444 / YUV422 / YCbCr422
            output_file: [str], dump file path
        '''
        if not(is_dump==''):
            self.config['output_setting']['is_dump'] = is_dump
        if not(dump_format==''):
            self.config['output_setting']['dump_format'] = dump_format
        if not(image_format==''):
            self.config['output_setting']['image_format'] = image_format    
        if not(output_file==''):
            self.config['output_setting']['output_file'] = output_file        
        return        

    def set_520_setting(self, radix='', bit_width='', rotate='',crop_fisrt='', round_w_to_16 ='',NUM_BANK_LINE='',BANK_ENTRY_CNT='',MAX_IMG_PREPROC_ROW_NUM='',MAX_IMG_PREPROC_COL_NUM=''):
        '''
        setting about 520 inproc

        Args:
            radix: [int], default 8
            bit_width: [int], default 8
            rotate: [int], 0 / 1 / 2, set rotate type
            crop_fisrt: [str], "yes" / "no", crop before inproc or not
            round_w_to_16: [str], "yes" / "no", round w align to 16 or not
            NUM_BANK_LINE: [int], default 32
            BANK_ENTRY_CNT: [int], default 512
            MAX_IMG_PREPROC_ROW_NUM: [int], default 511
            MAX_IMG_PREPROC_COL_NUM: [int], default 256
        '''
        if not(radix==''):
            self.config['520_setting']['radix'] = radix
        if not(bit_width==''):
            self.config['520_setting']['bit_width'] = bit_width        
        if not(rotate==''):
            self.config['520_setting']['rotate'] = rotate        
        if not(crop_fisrt==''):
            self.config['520_setting']['crop_fisrt'] = crop_fisrt
        if not(round_w_to_16==''):
            self.config['520_setting']['round_w_to_16'] = round_w_to_16 
        if not(NUM_BANK_LINE==''):
            self.config['520_setting']['NUM_BANK_LINE'] = NUM_BANK_LINE
        if not(BANK_ENTRY_CNT==''):
            self.config['520_setting']['BANK_ENTRY_CNT'] = BANK_ENTRY_CNT        
        if not(MAX_IMG_PREPROC_ROW_NUM==''):
            self.config['520_setting']['MAX_IMG_PREPROC_ROW_NUM'] = MAX_IMG_PREPROC_ROW_NUM        
        if not(MAX_IMG_PREPROC_COL_NUM==''):
            self.config['520_setting']['MAX_IMG_PREPROC_COL_NUM'] = MAX_IMG_PREPROC_COL_NUM 
        return

    def set_720_setting(self, radix='', bit_width='', rotate='',crop_fisrt='', matrix='',vector=''):
        '''
        setting about 720 inproc

        Args:
            radix: [int], default 8
            bit_width: [int], default 8
            rotate: [int], 0 / 1 / 2, set rotate type
            crop_fisrt: [str], "yes" / "no", crop before inproc or not
            matrix: [list]
            vector: [list]
        '''
        if not(radix==''):
            self.config['720_setting']['radix'] = radix
        if not(bit_width==''):
            self.config['720_setting']['bit_width'] = bit_width        
        if not(rotate==''):
            self.config['720_setting']['rotate'] = rotate        
        if not(crop_fisrt==''):
            self.config['720_setting']['crop_fisrt'] = crop_fisrt 
        return        

    def set_floating_setting(self, job_list = []):
        '''
        set_floating_setting, set floating pre-processing job list and order, can be combination of color, crop, resize, padding, normalize

        Args:
            job_list: [list], combination of "color" / "crop" / "resize" / "padding" / "normalize"
        '''
        if not(job_list==[]):
            self.config['floating_setting']['job_list'] = job_list 
        return

    def set_color_conversion(self, source_format = '', out_format='', simulation='', simulation_format=''):
        '''
        set_color_conversion

        setting about corlor conversion and inproc format unit.
        Turn simulation on can simulate rgb image to other image type.

        Args:
            source_format: [str], "rgb888" / "rgb565" / "yuv" / "ycbcr"
            out_format: [str], "rgb888" / "l" 
            simulation: [str], "yes" / "no"
            simulation_format: [str], "rgb565" / "yuv" / "ycbcr"
        '''  
        if not(source_format==''):
            self.config['function_setting']['color']['source_format'] = source_format
        if not(out_format==''):
            self.config['function_setting']['color']['out_format'] = out_format
        if not(simulation==''):
            self.config['function_setting']['color']['options']['simulation'] = simulation        
        if not(simulation_format==''):
            self.config['function_setting']['color']['options']['simulation_format'] = simulation_format    
 
        return
   
    def set_resize(self, type='',  keep_ratio='', calculate_ratio_using_CSim='',zoom='', resize_w='', resize_h = ''):
        '''
        set_resize, setting about resize and inproc resize unit.

        resize type can be bilinear or bilicubic as floating type, fixed or fixed_520 as fixed type.
        fixed_520 type has add some function to simulate 520 bug.

        Args:
            type[str]: "bilinear" / "bilicubic" / "cv2" / "fixed" / "fixed_520"
            keep_ratio[str]: "yes" / "no"
            calculate_ratio_using_CSim[str]: "yes" / "no" , calculate the ratio and scale using Csim function and C float
            zoom[str]: "yes" / "no", enable resize can zoom image or not
            resize_w[int]: if empty, then default will be model_size[0]
            resize_h[int]: if empty, then default will be model_size[0]
        '''
        if not(type==''):
            self.config['function_setting']['resize']['type'] = type
        if not(keep_ratio==''):
            self.config['function_setting']['resize']['keep_ratio'] = keep_ratio
        if not(calculate_ratio_using_CSim==''):
            self.config['function_setting']['resize']['calculate_ratio_using_CSim'] = calculate_ratio_using_CSim
        if not(zoom==''):
            self.config['function_setting']['resize']['zoom'] = zoom        
        if not(resize_w==''):
            self.config['function_setting']['resize']['resize_w'] = resize_w        
        if not(resize_h==''):
            self.config['function_setting']['resize']['resize_h'] = resize_h

        return

    def set_crop(self, type='',  crop_w='', crop_h='', start_x='', start_y='', end_x='', end_y='',align_w_to_4="",pad_square_to_4="",rounding_type=""):
        '''
        set_crop, setting about crop and rdma crop unit.

        crop type can be corner,center or specific.

        if type = corner and center, need to set crop_w and crop_h(or keep empty to set as model_size)
        
        if type = specific, need to set start_x, start_y, end_x and end_y
        
        if start_x, start_y, end_x and end_y all are not empty, then the type will turn to specific automatically
        
        Args:
        type: [str], "corner" / "center" / "specific"
        crop_w: [int], if empty, then default will be model_size[0]
        crop_h: [int], if empty, then default will be model_size[0]
        start_x: [int]
        start_y: [int]
        end_x: [int]
        end_y: [int]
        align_w_to_4: [str], crop length in w direction align to 4 or not
        pad_square_to_4: [str], pad to square(align 4) or not
        rounding_type: [int], 0-> x1,y1 take floor, x2,y2 take ceil; 1->all take rounding 
        '''
        if not(type==''):
            self.config['function_setting']['crop']['type'] = type
        if not(align_w_to_4==''):
            self.config['function_setting']['crop']['align_w_to_4'] = align_w_to_4
        if not(pad_square_to_4==''):
            self.config['function_setting']['crop']['pad_square_to_4'] = pad_square_to_4
        if not(rounding_type==''):
            self.config['function_setting']['crop']['rounding_type'] = rounding_type
        if not(crop_w==''):
            self.config['function_setting']['crop']['crop_w'] = crop_w
        if not(crop_h==''):
            self.config['function_setting']['crop']['crop_h'] = crop_h        
        if not(start_x==''):
            self.config['function_setting']['crop']['start_x'] = start_x
        if not(start_y==''):
            self.config['function_setting']['crop']['start_y'] = start_y
        if not(end_x==''):
            self.config['function_setting']['crop']['end_x'] = end_x
        if not(end_y==''):
            self.config['function_setting']['crop']['end_y'] = end_y
        return

    def set_padding(self, type='',  pad_val='',  padded_w='', padded_h='', pad_l='', pad_r='', pad_t='', pad_b=''):
        '''
        set_padding, setting about padding and inproc padding unit.

        crop type can be corner,center or specific.

        if type = corner and center, need to set out_w and out_h(or keep empty to set as model_size)
        
        if type = specific, need to set pad_l, pad_r, pad_t and pad_b
        
        if pad_l, pad_r, pad_t and pad_b all are not empty, then the type will turn to specific automatically

        if numerical type = 520 or 720, then the pad_val will adjust according radix automatically

        Args:
            type: [str], "corner" / "center" / "specific"
            pad_val: [float]
            out_w: [int]
            out_h: [int]
            pad_l: [int]
            pad_r: [int]
            pad_t: [int]
            pad_b: [int]
        '''
        if not(type==''):
            self.config['function_setting']['padding']['type'] = type
        if not(pad_val==''):
            self.config['function_setting']['padding']['pad_val'] = pad_val
        if not(padded_w==''):
            self.config['function_setting']['padding']['padded_w'] = padded_w
        if not(padded_h==''):
            self.config['function_setting']['padding']['padded_h'] = padded_h        
        if not(pad_l==''):
            self.config['function_setting']['padding']['pad_l'] = pad_l        
        if not(pad_r==''):
            self.config['function_setting']['padding']['pad_r'] = pad_r
        if not(pad_t==''):
            self.config['function_setting']['padding']['pad_t'] = pad_t
        if not(pad_b==''):
            self.config['function_setting']['padding']['pad_b'] = pad_b
        return

    def set_normalize(self, type='',  scale='',  bias='', mean='', std =''):
        '''
        set_normalize, setting about normalize and inproc chen unit.

        if numerical type = floating:
        normalize type can be customized, torch, tf, caffe, yolo or kneron
        if type = customized, need to set scale, bias, mean and std

        if numerical type = 520 or 720:
        normalize type can be tf, yolo or kneron

        Args:
            type: [str], "customized" / "torch" / "tf" / "caffe" / "yolo" / "kneron"
            scale: [float]
            bias: [float]
            mean: [list,3]
            std: [list,3]
        '''
        if not(type==''):
            self.config['function_setting']['normalize']['type'] = type
        if not(scale==''):
            self.config['function_setting']['normalize']['scale'] = scale
        if not(bias==''):
            self.config['function_setting']['normalize']['bias'] = bias
        if not(mean==''):
            self.config['function_setting']['normalize']['mean'] = mean      
        if not(std==''):
            self.config['function_setting']['normalize']['std'] = std           
        return

    def load_image(self, image, is_raw = False , raw_img_type = '', raw_img_fmt = '', img_in_height = 0, img_in_width = 0):
        '''
        load_image function

        Args:
            image: [np.array/str], can be np.array or file path(bin/hex/jpg)
            is_raw: [bool], is raw image or not (bin or hex)
            raw_img_type: [str], "bin" / "hex"
            raw_img_fmt: [str], "yuv444" / "ycbcr444" / "yuv422" / "ycbcr422" / "rgb565" / "nir"
            img_in_width: [int]
            img_in_height: [int]
        
        Returns:
            out: [np.array], not include color convert
        '''
        if isinstance(image, np.ndarray):
            return image
        if str2bool(is_raw):
            dic ={}
            dic['raw_img_fmt'] = raw_img_fmt
            dic['img_in_height'] = img_in_height
            dic['img_in_width'] = img_in_width
            if raw_img_type.lower() in ['bin','BIN']:
                image_data = bin_loader(image,**dic)
            elif raw_img_type.lower() in ['hex','HEX']:
                image_data = hex_loader(image,**dic)
        elif isinstance(image, str):
            image = Image.open(image).convert("RGB")
            image_data = np.array(image).astype('uint8')
         
        assert isinstance(image_data, np.ndarray)
        return image_data

    def dump_image(self,image_data):
        '''
        dump_image function, according config setting to dump image, txt, bin or hex

        Args:
            image: [np.array]
        '''
        assert isinstance(image_data, np.ndarray)
        assert (len(image_data.shape) >= 2)

        if (len(image_data.shape) == 2):
            source_format = 'L'
        if (image_data.shape[2] == 4):
            source_format = 'RGBA8888'
        else:
            source_format = 'RGB888'

        convert = ColorConversion.runner()
        if (source_format == 'L') & (self.config['output_setting']['image_format'].lower() not in ['L', 'l', 'NIR', 'nir']):
            convert.update(**{"source_format": "L","out_format": "RGB888"})
            image_data, _ = convert.run(image_data)
            source_format = 'RGB888'

        if (source_format == 'RGBA8888') & (self.config['output_setting']['image_format'].lower() not in ['RGBA8888', 'rgba8888','RGBA','rgba']):
            convert.update(**{"source_format": "RGBA8888","out_format": "RGB888"})
            image_data, _ = convert.run(image_data)
            source_format = 'RGB888'

    
        if (self.config['output_setting']['image_format'].lower() in ['RGB565', 'rgb565']):
            convert.update(**{"source_format": source_format,"out_format": "RGB565"})
            image_data_565, _ = convert.run(image_data)
            image_data = np.zeros((image_data_565.shape[0],image_data_565.shape[1],2), dtype=np.uint8)
            image_data[:,:,1] = ( image_data_565[:,:,0] << 3 ) + ( image_data_565[:,:,1] >> 3 )
            image_data[:,:,0] = ( (image_data_565[:,:,1] & 0x07) << 5 ) + image_data_565[:,:,2]
        elif (self.config['output_setting']['image_format'].lower() in ['RGBA8888', 'rgba8888','RGBA','rgba']) & (source_format != 'RGBA8888'):
            convert.update(**{"source_format": source_format,"out_format": "rgba"})
            image_data, _ = convert.run(image_data)
        elif (self.config['output_setting']['image_format'].lower() in ['L', 'l', 'NIR', 'nir']):
            convert.update(**{"source_format": source_format,"out_format": "L"})
            image_data, _ = convert.run(image_data)
        elif (self.config['output_setting']['image_format'].lower() in['YUV', 'YUV444','yuv','yuv444']):
            convert.update(**{"source_format": source_format,"out_format": "YUV444"})
            image_data_YUV, _ = convert.run(image_data)
            image_data = np.zeros((image_data_YUV.shape[0],image_data_YUV.shape[1],4), dtype=np.uint8)
            image_data[:,:,3] = image_data_YUV[:,:,0]
            image_data[:,:,2] = image_data_YUV[:,:,1]
            image_data[:,:,1] = image_data_YUV[:,:,2]
        elif (self.config['output_setting']['image_format'].lower() in['YUV422','yuv422']):
            convert.update(**{"source_format": source_format,"out_format": "YUV444"})
            image_data_YUV, _ = convert.run(image_data)
            pixels = image_data_YUV.shape[0] * image_data_YUV.shape[1]
            image_data = np.zeros((pixels*2,1), dtype=np.uint8)
            image_data_YUV = image_data_YUV.reshape((-1,1))
            for i in range(0,image_data.shape[0],4): 
                j = i//2 #source index
                image_data[i+3,0] = image_data_YUV[j*3,0]
                image_data[i+2,0] = image_data_YUV[j*3+1,0]
                image_data[i+1,0] = image_data_YUV[j*3+3,0]
                image_data[i,0] = image_data_YUV[j*3+5,0]
        elif (self.config['output_setting']['image_format'].lower() in['YCBCR', 'YCBCR444','YCbCr','YCbCr444','ycbcr','ycbcr444']):
            convert.update(**{"source_format": source_format,"out_format": "YCBCR444"})
            image_data_YCBCR, _ = convert.run(image_data)
            image_data = np.zeros((image_data_YCBCR.shape[0],image_data_YCBCR.shape[1],4), dtype=np.uint8)
            image_data[:,:,3] = image_data_YCBCR[:,:,0]
            image_data[:,:,2] = image_data_YCBCR[:,:,1]
            image_data[:,:,1] = image_data_YCBCR[:,:,2]
        elif (self.config['output_setting']['image_format'].lower() in['YCBCR422','YCbCr422','ycbcr422']):
            convert.update(**{"source_format": source_format,"out_format": "YCBCR422"})
            image_data_YCBCR, _ = convert.run(image_data)
            image_data = np.zeros((image_data_YCBCR.shape[0],image_data_YCBCR.shape[1],2), dtype=np.uint8)
            pixels = image_data_YCBCR.shape[0] * image_data_YCBCR.shape[1]
            image_data = np.zeros((pixels*2,1), dtype=np.uint8)
            image_data_YCBCR = image_data_YCBCR.reshape((-1,1))
            for i in range(0,image_data.shape[0],4): 
                j = i//2 #source index
                image_data[i+3,0] = image_data_YCBCR[j*3,0]
                image_data[i+2,0] = image_data_YCBCR[j*3+1,0]
                image_data[i+1,0] = image_data_YCBCR[j*3+3,0]
                image_data[i,0] = image_data_YCBCR[j*3+5,0]
  
        if self.config['output_setting']['dump_format'].lower() in ['txt', 'TXT']:
            np.savetxt(self.config['output_setting']['output_file'],image_data.reshape((-1,1)),fmt="%.8f")
        elif self.config['output_setting']['dump_format'].lower() in ['bin', 'BIN']:
            image_data.reshape((-1,1)).astype("uint8").tofile(self.config['output_setting']['output_file'])
        elif self.config['output_setting']['dump_format'].lower() in ['hex', 'HEX']:
            height, width, c = image_data.shape
            output_line = math.floor((height * width) / 4)
            image_f = image_data.reshape((height * width, c))
            f = open(self.config['output_setting']['output_file'], "w")
            for i in range(output_line):
                pixels = ""
                for j in range(min((i+1)*4-1, image_f.shape[0]-1), i*4-1, -1):
                    pixels = pixels + str_fill(hex(image_f[j, 3]).lstrip("0x"))
                    pixels = pixels + str_fill(hex(image_f[j, 2]).lstrip("0x"))
                    pixels = pixels + str_fill(hex(image_f[j, 1]).lstrip("0x"))
                    pixels = pixels + str_fill(hex(image_f[j, 0]).lstrip("0x"))
                f.write(pixels + "\n")
        return

    def run_whole_process(self, image):
        '''
        run_whole_process, according config setting to run all pre-processing

        Args:
            image: [np.array/str], can be np.array or file path(bin/jpg)

        Returns:
            out: [np.array]
        '''
        assert (self.error_state == 0)

        image_data = self.load_image(
            image, 
            is_raw = self.config['raw_img']["is_raw_img"], 
            raw_img_type = self.config['raw_img']["raw_img_type"], 
            raw_img_fmt = self.config['raw_img']["raw_img_fmt"],
            img_in_height= self.config['raw_img']["img_in_height"], 
            img_in_width=self.config['raw_img']["img_in_width"])

        if str2bool(self.config['raw_img']["is_raw_img"]):
            self.set_color_conversion(source_format=self.config['raw_img']["raw_img_fmt"])
        elif isinstance(image, str):
            self.set_color_conversion(source_format='RGB888')

        h_ori = image_data.shape[0]     
        w_ori = image_data.shape[1] 
        
        if self.config['numerical_type'] == 'floating':
            image_data = self.__run_whole_process_floating(image_data)
        elif self.config['numerical_type'] == '520':
            image_data = self.__run_whole_process_520(image_data)
        elif self.config['numerical_type'] == '720':
            image_data = self.__run_whole_process_720(image_data)
       
        if str2bool(self.config['output_setting']['is_dump']):
            self.dump_image(image_data)
        
        scale = max(1.0*w_ori / image_data.shape[1], 1.0*h_ori / image_data.shape[0])
        out = {'h_ori': h_ori, 'w_ori': w_ori, "scale": scale}
        return image_data, out

    def __run_whole_process_floating(self,image_data):
        '''
        private function
        '''
        for job in self.config['floating_setting']['job_list']:
            if job.lower() in ['crop','color','resize','normalize','padding']:
                image_data, _ = self.funcs[job](image_data)

        return image_data

    def __run_whole_process_520(self,image_data):
        '''
        private function
        '''
        # init from config
        originH, originW, _ = image_data.shape
        npu_img_w = self.config['model_size'][0]
        npu_img_h = self.config['model_size'][1]

        if self.config['function_setting']['padding']['type'].lower() in ['center','CENTER','Center','0',0]:
            pad_mode = 0
        elif self.config['function_setting']['padding']['type'].lower() in ['corner','CORNER','Corner','1',1]:
            pad_mode = 1
        else:
            pad_mode = 2

        if not str2bool(self.config['function_setting']['resize']['keep_ratio']):
            pad_mode = 2

        NUM_BANK_LINE = self.config['520_setting']['NUM_BANK_LINE']
        BANK_ENTRY_CNT = self.config['520_setting']['BANK_ENTRY_CNT']
        MAX_IMG_PREPROC_ROW_NUM = self.config['520_setting']['MAX_IMG_PREPROC_ROW_NUM']
        MAX_IMG_PREPROC_COL_NUM = self.config['520_setting']['MAX_IMG_PREPROC_COL_NUM']
        
        raw_fmt = self.config['function_setting']['color']['source_format']
        crop_fisrt = str2bool(self.config['520_setting']['crop_fisrt'])
        keep_ratio = str2bool(self.config['function_setting']['resize']['keep_ratio'])

        # init crop
        if crop_fisrt:
            startW = self.config['function_setting']['crop']['start_x'] 
            startH = self.config['function_setting']['crop']['start_y'] 
            cropW = self.config['function_setting']['crop']['end_x'] - self.config['function_setting']['crop']['start_x'] 
            cropH = self.config['function_setting']['crop']['end_y'] - self.config['function_setting']['crop']['start_y'] 
        else:
            startW = 0
            startH = 0
            cropW = originW
            cropH = originH

        crop_num = [0] * 4
        crop_num[0] = startW                      #left
        crop_num[1] = startH                      #top
        crop_num[2] = originW - (startW + cropW)  #right
        crop_num[3] = originH - (startH + cropH)  #bottom
        
        # calculate scaleW scaleH padW padH 
        if keep_ratio:
            out_w = npu_img_w
            out_h = npu_img_h
            orig_w = cropW
            orig_h = cropH
            
            w_ratio = c_float(out_w * 1.0 / (orig_w * 1.0)).value
            h_ratio = c_float(out_h * 1.0 / (orig_h * 1.0)).value
            scale_ratio = 0.0
            scale_target_w = 0
            scale_target_h = 0
            padH = 0
            padW = 0

            bScaleW = True if w_ratio < h_ratio else False
            if bScaleW:
                scale_ratio = w_ratio
                scale_target_w = int(c_float(scale_ratio * orig_w + 0.5).value)
                scale_target_h = int(c_float(scale_ratio * orig_h + 0.5).value)
                assert (abs(scale_target_w - out_w) <= 1), "Error: scale down width cannot meet expectation\n"
                padH = out_h - scale_target_h
                padW = 0
                assert (padH >= 0), "Error: padH shouldn't be less than zero\n"
            else:
                scale_ratio = h_ratio 
                scale_target_w = int(c_float(scale_ratio * orig_w + 0.5).value)
                scale_target_h = int(c_float(scale_ratio * orig_h + 0.5).value)
                assert (abs(scale_target_h - out_h) <= 1), "Error: scale down height cannot meet expectation\n"
                padW = out_w - scale_target_w
                padH = 0
                assert (padW >= 0), "Error: padW shouldn't be less than zero\n"
                
            scaleW = out_w - padW
            scaleH = out_h - padH
        else:
            scaleW = npu_img_w
            scaleH = npu_img_h
            padW = 0
            padH = 0
        
        # calculate pad_top pad_bottom pad_left pad_right 
        if (pad_mode == 0):
            # pad on both side
            pad_top = padH // 2
            pad_bottom = (padH // 2) + (padH % 2)
            pad_left = padW // 2
            pad_right = (padW // 2) + (padW % 2)
        elif (pad_mode == 1):
            # only pad right and bottom
            pad_top = 0
            pad_bottom = padH
            pad_left = 0
            pad_right = padW
        else:
            pad_top = 0
            pad_bottom = 0
            pad_left = 0
            pad_right = 0

        if (pad_right > 127 or pad_bottom > 127):
            print("Pad value larger than 127 is not supported\n")

        orig_pad_num = [0] * 4
        orig_pad_num[0] = pad_left
        orig_pad_num[1] = pad_top
        orig_pad_num[2] = pad_right
        orig_pad_num[3] = pad_bottom

        valid_in_row = cropH
        valid_in_col = cropW
        out_row = scaleH + padH
        out_col = scaleW + padW
    
        # calculate cut_total
        max_row = int(math.floor(BANK_ENTRY_CNT * NUM_BANK_LINE / (out_col / 4)))
        max_row = min(max_row, MAX_IMG_PREPROC_ROW_NUM)
        
        if (pad_mode == 0):
            big_pad_row = (out_row % max_row) < (pad_bottom + 4)
            if (big_pad_row):
                last_row = int(pad_bottom + 4)
                cut_total = int(math.ceil( float(out_row - last_row) / max_row) + 1)
            else:
                cut_total = int(math.ceil( float(out_row) / max_row))
        elif (pad_mode == 1):
            big_pad_row = (out_row % max_row) < (pad_bottom + 4)
            last_row = max_row
            if (big_pad_row):
                cut_total = int(math.ceil( float(out_row - last_row) / max_row) + 1)
            else:
                cut_total = int(math.ceil( float(out_row) / max_row))
        else:
            big_pad_row = False
            cut_total = int(math.ceil( float(out_row) / max_row))

        # calculate seg_cnt
        max_col = MAX_IMG_PREPROC_COL_NUM
        last_col = 0
        if (out_col % max_col):
            if (pad_mode == 0):
                big_pad_col = (out_col % max_col) < (pad_right + 4)
                if (big_pad_col):
                    last_col = round_up_n(pad_right + 4, 4)
                    seg_cnt = math.ceil( float(out_col - last_col) / max_col) + 1
                else:
                    seg_cnt = math.ceil( float(out_col) / max_col)
            elif (pad_mode == 1):
                big_pad_col = (out_col % max_col) < (pad_right + 4)
                last_col = max_col
                if (big_pad_col):
                    seg_cnt = math.ceil( float(out_col - last_col) / max_col) + 1
                else:
                    seg_cnt = math.ceil( float(out_col) / max_col)
            else:
                big_pad_col = False
                seg_cnt = math.ceil( float(out_col) / max_col)
        else:
            big_pad_col = False
            seg_cnt = math.ceil( float(out_col) / max_col)
            
        # start loop
        if (big_pad_row):
            remain_row = out_row - last_row
        else:
            remain_row = out_row
        start_row = 0
        row_num = 0
        for r in range(0, cut_total):
            start_row += row_num
            block_start_row = cal_img_row_offset(crop_num, orig_pad_num, start_row, out_row, originH)
            if (big_pad_row) and (r == (cut_total - 1)):
                row_num = last_row
            else:
                row_num = min(max_row, remain_row)
            
            # due to HW only support max col = 256, we may need to process data in segments */
            if(big_pad_col):
                remain_col =  (out_col - last_col)
            else:
                remain_col = out_col
            start_col = 0
            col_num = 0
            block_start_col = crop_num[0]
            block_col = 0
            for c in range(0,seg_cnt):
                start_col += col_num
                block_start_col += block_col
                if (big_pad_col) and (c == (seg_cnt - 1)):
                    col_num = last_col
                else:
                    col_num = min(remain_col, MAX_IMG_PREPROC_COL_NUM)
                
                pad_num = get_pad_num(orig_pad_num, (c == 0), (r == 0), (c == seg_cnt - 1), (r == cut_total - 1))
                block_row = int(valid_in_row * (row_num - pad_num[1] - pad_num[3]) / (out_row - orig_pad_num[1] - orig_pad_num[3]))
                block_col = int(valid_in_col * (col_num - pad_num[0] - pad_num[2]) / (out_col - orig_pad_num[0] - orig_pad_num[2]))
                #/* (src_w * byte_per_pixel) should align to multiple of 4-byte and 2 cols */
                byte_per_pixel = get_byte_per_pixel(raw_fmt)
                new_block_col = round_up_n(round_up_n(block_col, (4 / byte_per_pixel)), 2)

                if (new_block_col > block_col):
                    if byte_per_pixel == 1:
                        block_col = new_block_col - 4
                    elif byte_per_pixel == 4:
                        block_col = new_block_col - 2
                    else:
                        block_col = new_block_col - 2
    
                ##
                # crop
                self.set_crop(start_x=block_start_col, start_y=block_start_row, end_x=block_start_col+block_col,end_y=block_start_row+block_row,align_w_to_4=False)
                image_temp, _ = self.funcs['crop'](image_data)

                # color
                image_temp, _ = self.funcs['color'](image_temp)

                # resize
                self.set_resize(type='fixed_520',keep_ratio='no',calculate_ratio_using_CSim = 'yes', resize_w=(col_num - pad_num[0] - pad_num[2]),resize_h=(row_num - pad_num[1] - pad_num[3]))
                image_temp, _ = self.funcs['resize'](image_temp)

                # normalize
                image_temp, _ = self.funcs['normalize'](image_temp)

                # padding
                self.set_padding(type='specific',pad_l=pad_num[0],pad_t=pad_num[1],pad_r=pad_num[2],pad_b=pad_num[3])
                image_temp, _ = self.funcs['padding'](image_temp)

                ##
                remain_col -= col_num
                if c == 0:
                    image_temp_H = image_temp
                else:
                    image_temp_H = np.concatenate((image_temp_H, image_temp), axis=1)

            ##
            remain_row -= row_num
            if r == 0:
                image_temp_V = image_temp_H
            else:
                image_temp_V = np.concatenate((image_temp_V, image_temp_H), axis=0)

        ##
        image_data = image_temp_V

        # # round_w_to_16
        if str2bool(self.config['520_setting']['round_w_to_16']):
            out_w_16 = round_up_n(out_col,16)
            image = np.ones((out_row,out_w_16 - out_col,4)) *128
            image_data = np.concatenate((image_data, image), axis=1)
        
        # rotate
        rotate = self.config['520_setting']['rotate']
        if not (rotate == 0):
            dic = {}
            dic['rotate_direction'] = rotate
            rotate = Rotate.runner(**dic, b_print = str2bool(self.config['print_info']))
            image_data = rotate.run(image_data)

        return image_data

    def __run_whole_process_720(self,image_data):
        '''
        private function
        '''
        # init from config
        crop_fisrt = str2bool(self.config['720_setting']['crop_fisrt'])
        matrix_c00 = self.config['720_setting']['matrix_c00']
        matrix_c01 = self.config['720_setting']['matrix_c01']
        matrix_c02 = self.config['720_setting']['matrix_c02']
        matrix_c10 = self.config['720_setting']['matrix_c10']
        matrix_c11 = self.config['720_setting']['matrix_c11']
        matrix_c12 = self.config['720_setting']['matrix_c12']
        matrix_c20 = self.config['720_setting']['matrix_c20']
        matrix_c21 = self.config['720_setting']['matrix_c21']
        matrix_c22 = self.config['720_setting']['matrix_c22']
        vector_b00 = self.config['720_setting']['vector_b00']
        vector_b01 = self.config['720_setting']['vector_b01']
        vector_b02 = self.config['720_setting']['vector_b02']
        shiftvalue = self.config['720_setting']['shift']
        subvalue = self.config['720_setting']['sub']

        #crop
        if crop_fisrt:
            image_data, _ = self.funcs['crop'](image_data)

        #color
        image_data, _ = self.funcs['color'](image_data)

        #resize
        self.set_resize(type='fixed_720',calculate_ratio_using_CSim = 'yes')
        image_data, _ = self.funcs['resize'](image_data)

        #matrix
        h, w, c = image_data.shape
        image_f = image_data.reshape((h * w, c))
        matrix_c = np.array([[matrix_c00, matrix_c01, matrix_c02],
                             [matrix_c10, matrix_c11, matrix_c12],
                             [matrix_c20, matrix_c21, matrix_c22]])
        b = np.array([[vector_b00], [vector_b01], [vector_b02]])
        calculated_image_f = np.zeros(image_f.shape, dtype=np.uint8)
        for i in range(h*w):
            pt = np.swapaxes(image_f[np.newaxis, i, :], 0, 1)
            matrix_pt = np.floor(np.multiply((matrix_c @ pt), 1/np.power(2, 1)))
            matrix_pt.astype(int)
            result = np.floor(np.multiply(np.add(matrix_pt, b), 1/np.power(2, 7)))
            result.astype(int)

            result = twos_complement_pix(result)

            if shiftvalue == 1:
                result = clip_pix(np.add(result, -128 * np.ones(result.shape)), -128, 127)
            else:
                result = clip_pix(result, 0, 255)

            result = result + np.array([[subvalue], [subvalue], [subvalue]])
            calculated_image_f[i, :] = clip_ary(np.squeeze(result))

        image_data = calculated_image_f.reshape(image_data[:, :, 0:3].shape)

        #padding
        image_data, _ = self.funcs['padding'](image_data)

        return image_data
    
    def run_crop(self, image_data):
        '''
        @brief
        run_crop, according config setting to run crop

        @param
        image[np.array] : only can be np.array

        @return
        np.array
        '''
        self.__update_crop()
        image_data, info = self.subclass['crop'].run(image_data)
        return image_data, info

    def run_color_conversion(self, image_data):
        '''
        @brief
        run_color_conversion, according config setting to run color conversion

        @param
        image[np.array] : only can be np.array

        @return
        np.array
        '''
        self.__update_color()
        image_data, info = self.subclass['color'].run(image_data)
        return image_data,info

    def run_resize(self, image_data):
        '''
        @brief
        run_resize, according config setting to run resize

        @param
        image[np.array] : only can be np.array

        @return
        np.array
        '''
        self.__update_resize()
        image_data,info = self.subclass['resize'].run(image_data)
        return image_data,info

    def run_normalize(self, image_data):
        '''
        @brief
        run_normalize, according config setting to run normalize

        @param
        image[np.array] : only can be np.array

        @return
        np.array
        '''
        self.__update_normalize()
        image_data,info = self.subclass['normalize'].run(image_data)
        return image_data,info
    
    def run_padding(self, image_data):
        '''
        @brief
        run_padding, according config setting to run padding

        @param
        image[np.array] : only can be np.array

        @return
        np.array
        '''
        self.__update_padding()
        image_data,info = self.subclass['padding'].run(image_data)
        return image_data,info
        
        
