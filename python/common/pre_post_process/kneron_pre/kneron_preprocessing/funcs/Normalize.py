import numpy as np
from .utils import str2bool, str2int, str2float, clip_ary

class runner(object):
    def __init__(self):
        self.set = {
            'general': {
                'print_info':'no',
                'model_size':[0,0],
                'numerical_type':'floating',
                'type': 'kneron'
            },
            'floating':{
                "scale": 1,
                "bias": 0,
                "mean": "",
                "std": "",
            },
            'hw':{
                "radix":8,
                "shift":"",
                "sub":""
            }
        }
        return

    def update(self, **kwargs):
        #
        self.set.update(kwargs)

        #
        if self.set['general']['numerical_type'] == '520':
            if self.set['general']['type'].lower() in ['TF', 'Tf', 'tf']:
                self.fun_normalize = self._chen_520
                self.shift = 7 - self.set['hw']['radix']
                self.sub = 128
            elif self.set['general']['type'].lower() in ['YOLO', 'Yolo', 'yolo']:
                self.fun_normalize = self._chen_520
                self.shift = 8 - self.set['hw']['radix']
                self.sub = 0
            elif self.set['general']['type'].lower() in ['KNERON', 'Kneron', 'kneron']:
                self.fun_normalize = self._chen_520
                self.shift = 8 - self.set['hw']['radix']
                self.sub = 128
            else:
                self.fun_normalize = self._chen_520
                self.shift = 0
                self.sub = 0      
        elif self.set['general']['numerical_type'] == '720':
                self.fun_normalize = self._chen_720
                self.shift = 0
                self.sub = 0                   
        else:
            if self.set['general']['type'].lower() in ['TORCH', 'Torch', 'torch']:
                self.fun_normalize = self._normalize_torch
                self.set['floating']['scale'] = 255.
                self.set['floating']['mean'] = [0.485, 0.456, 0.406]
                self.set['floating']['std'] = [0.229, 0.224, 0.225]
            elif self.set['general']['type'].lower() in ['TF', 'Tf', 'tf']:
                self.fun_normalize = self._normalize_tf
                self.set['floating']['scale'] = 127.5
                self.set['floating']['bias'] = -1.
            elif self.set['general']['type'].lower() in ['CAFFE', 'Caffe', 'caffe']:
                self.fun_normalize = self._normalize_caffe
                self.set['floating']['mean'] = [103.939, 116.779, 123.68]
            elif self.set['general']['type'].lower() in ['YOLO', 'Yolo', 'yolo']:
                self.fun_normalize = self._normalize_yolo
                self.set['floating']['scale'] = 255.
            elif self.set['general']['type'].lower() in ['KNERON', 'Kneron', 'kneron']:
                self.fun_normalize = self._normalize_kneron
                self.set['floating']['scale'] = 256.
                self.set['floating']['bias'] = -0.5
            else:
                self.fun_normalize = self._normalize_customized
                self.set['floating']['scale'] = str2float(self.set['floating']['scale'])
                self.set['floating']['bias'] = str2float(self.set['floating']['bias'])
                if self.set['floating']['mean'] != None:
                    if len(self.set['floating']['mean']) != 3:
                        self.set['floating']['mean'] = None
                if self.set['floating']['std'] != None:
                    if len(self.set['floating']['std']) != 3:
                        self.set['floating']['std'] = None


    def print_info(self):
        if self.set['general']['numerical_type'] == '520':
            print("<normalize>",
            'numerical_type', self.set['general']['numerical_type'],
            ", type:", self.set['general']['type'],
            ', shift:',self.shift, 
            ', sub:', self.sub)
        else:
            print("<normalize>",
            'numerical_type', self.set['general']['numerical_type'],
            ", type:", self.set['general']['type'],
            ', scale:',self.set['floating']['scale'], 
            ', bias:', self.set['floating']['bias'],
            ', mean:', self.set['floating']['mean'],
            ', std:',self.set['floating']['std'])

    def run(self, image_data):
        # print info
        if str2bool(self.set['general']['print_info']):
            self.print_info()

        # norm
        image_data = self.fun_normalize(image_data)

        # output
        info = {}
        return image_data, info

    def _normalize_torch(self, x):
        if len(x.shape) != 3:
            return x
        x = x.astype('float')
        x = x / self.set['floating']['scale']
        x[..., 0] -= self.set['floating']['mean'][0]
        x[..., 1] -= self.set['floating']['mean'][1]
        x[..., 2] -= self.set['floating']['mean'][2]
        x[..., 0] /= self.set['floating']['std'][0]
        x[..., 1] /= self.set['floating']['std'][1]
        x[..., 2] /= self.set['floating']['std'][2]
        return x

    def _normalize_tf(self, x):
        # print('_normalize_tf')
        x = x.astype('float')
        x = x / self.set['floating']['scale']
        x = x + self.set['floating']['bias']
        return x

    def _normalize_caffe(self, x):
        if len(x.shape) != 3:
            return x
        x = x.astype('float')
        x = x[..., ::-1]
        x[..., 0] -= self.set['floating']['mean'][0]
        x[..., 1] -= self.set['floating']['mean'][1]
        x[..., 2] -= self.set['floating']['mean'][2]
        return x

    def _normalize_yolo(self, x):
        # print('_normalize_yolo')
        x = x.astype('float')
        x = x / self.set['floating']['scale']
        return x

    def _normalize_kneron(self, x):
        # print('_normalize_kneron')
        x = x.astype('float')
        x = x/self.set['floating']['scale']
        x = x + self.set['floating']['bias']
        return x

    def _normalize_customized(self, x):
        # print('_normalize_customized')
        x = x.astype('float')
        if  self.set['floating']['scale'] != 0:
            x = x/ self.set['floating']['scale'] 
        x = x + self.set['floating']['bias'] 
        if self.set['floating']['mean'] is not None:
            x[..., 0] -= self.set['floating']['mean'][0]
            x[..., 1] -= self.set['floating']['mean'][1]
            x[..., 2] -= self.set['floating']['mean'][2]
        if self.set['floating']['std'] is not None:
            x[..., 0] /= self.set['floating']['std'][0]
            x[..., 1] /= self.set['floating']['std'][1]
            x[..., 2] /= self.set['floating']['std'][2]

        return x

    def _chen_520(self, x):
        # print('_chen_520')
        x = (x - self.sub).astype('uint8')
        x = (np.right_shift(x,self.shift))
        x=x.astype('uint8')
        return x

    def _chen_720(self, x):
        # print('_chen_720')
        if self.shift == 1:
            x = x + np.array([[self.sub], [self.sub], [self.sub]])
        else:
            x = x + np.array([[self.sub], [self.sub], [self.sub]])
        return x