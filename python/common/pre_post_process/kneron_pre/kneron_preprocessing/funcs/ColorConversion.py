import numpy as np
from PIL import Image
from .utils import signed_rounding, clip, str2bool

format_bit = 10
c00_yuv = 1
c02_yuv = 1436
c10_yuv = 1
c11_yuv = -354
c12_yuv = -732
c20_yuv = 1
c21_yuv = 1814
c00_ycbcr = 1192
c02_ycbcr = 1634
c10_ycbcr = 1192
c11_ycbcr = -401
c12_ycbcr = -833
c20_ycbcr = 1192
c21_ycbcr = 2065

Matrix_ycbcr_to_rgb888 = np.array(
    [[1.16438356e+00,  1.16438356e+00,  1.16438356e+00],
     [2.99747219e-07, - 3.91762529e-01,  2.01723263e+00],
     [1.59602686e+00, - 8.12968294e-01,  3.04059479e-06]])

Matrix_rgb888_to_ycbcr = np.array(
    [[0.25678824, - 0.14822353,  0.43921569],
     [0.50412941, - 0.29099216, - 0.36778824],
     [0.09790588,  0.43921569, - 0.07142745]])

Matrix_rgb888_to_yuv = np.array(
    [[ 0.29899106, -0.16877996,  0.49988381],
    [ 0.5865453,  -0.33110385, -0.41826072],
    [ 0.11446364,  0.49988381, -0.08162309]])

# Matrix_rgb888_to_yuv = np.array(
#     [[0.299, - 0.147,   0.615],
#      [0.587, - 0.289, - 0.515],
#      [0.114,   0.436, - 0.100]])

# Matrix_yuv_to_rgb888 = np.array(
#     [[1.000,   1.000,  1.000],
#      [0.000, - 0.394,  2.032],
#      [1.140, - 0.581,  0.000]])

class runner(object):
    def __init__(self):
        self.set = {
            'print_info':'no',
            'model_size':[0,0],
            'numerical_type':'floating',
            "source_format": "rgb888",
            "out_format": "rgb888",
            "options": {
                "simulation": "no",
                "simulation_format": "rgb888"
            }
        }

    def update(self, **kwargs):
        #
        self.set.update(kwargs)

        ## simulation
        self.funs = []
        if str2bool(self.set['options']['simulation']) and self.set['source_format'].lower() in ['RGB888', 'rgb888', 'RGB', 'rgb']:
            if self.set['options']['simulation_format'].lower() in ['YUV422', 'yuv422', 'YUV', 'yuv']:
                self.funs.append(self._ColorConversion_RGB888_to_YUV422)
                self.set['source_format'] = 'YUV422'
            elif self.set['options']['simulation_format'].lower() in ['YCBCR422', 'YCbCr422', 'ycbcr422', 'YCBCR', 'YCbCr', 'ycbcr']:
                self.funs.append(self._ColorConversion_RGB888_to_YCbCr422)
                self.set['source_format'] = 'YCbCr422'
            elif self.set['options']['simulation_format'].lower() in['RGB565', 'rgb565']:
                self.funs.append(self._ColorConversion_RGB888_to_RGB565)
                self.set['source_format'] = 'RGB565'
        
        ## to rgb888
        if self.set['source_format'].lower() in ['YUV444', 'yuv444','YUV422', 'yuv422', 'YUV', 'yuv']:
            self.funs.append(self._ColorConversion_YUV_to_RGB888)
        elif self.set['source_format'].lower() in ['YCBCR444', 'YCbCr444', 'ycbcr444','YCBCR422', 'YCbCr422', 'ycbcr422', 'YCBCR', 'YCbCr', 'ycbcr']:
            self.funs.append(self._ColorConversion_YCbCr_to_RGB888)
        elif self.set['source_format'].lower() in ['RGB565', 'rgb565']:
            self.funs.append(self._ColorConversion_RGB565_to_RGB888)
        elif self.set['source_format'].lower() in ['l', 'L' , 'nir', 'NIR']:
            self.funs.append(self._ColorConversion_L_to_RGB888)
        elif self.set['source_format'].lower() in ['RGBA8888', 'rgba8888' , 'RGBA', 'rgba']:
            self.funs.append(self._ColorConversion_RGBA8888_to_RGB888)

        ## output format
        if self.set['out_format'].lower() in ['L', 'l']:
            self.funs.append(self._ColorConversion_RGB888_to_L)
        elif self.set['out_format'].lower() in['RGB565', 'rgb565']:
            self.funs.append(self._ColorConversion_RGB888_to_RGB565)
        elif self.set['out_format'].lower() in['RGBA', 'RGBA8888','rgba','rgba8888']:
            self.funs.append(self._ColorConversion_RGB888_to_RGBA8888)
        elif self.set['out_format'].lower() in['YUV', 'YUV444','yuv','yuv444']:
            self.funs.append(self._ColorConversion_RGB888_to_YUV444)
        elif self.set['out_format'].lower() in['YUV422','yuv422']:
            self.funs.append(self._ColorConversion_RGB888_to_YUV422)
        elif self.set['out_format'].lower() in['YCBCR', 'YCBCR444','YCbCr','YCbCr444','ycbcr','ycbcr444']:
            self.funs.append(self._ColorConversion_RGB888_to_YCbCr444)
        elif self.set['out_format'].lower() in['YCBCR422','YCbCr422','ycbcr422']:
            self.funs.append(self._ColorConversion_RGB888_to_YCbCr422)

    def print_info(self):
        print("<colorConversion>",
              "source_format:", self.set['source_format'],
              ', out_format:', self.set['out_format'],
              ', simulation:', self.set['options']['simulation'],
              ', simulation_format:', self.set['options']['simulation_format'])

    def run(self, image_data):
        assert isinstance(image_data, np.ndarray)
        # print info
        if str2bool(self.set['print_info']):
            self.print_info()

        # color
        for _, f in enumerate(self.funs):
            image_data = f(image_data)

        # output
        info = {}
        return image_data, info

    def _ColorConversion_RGB888_to_YUV444(self, image):
        ## floating
        image = image.astype('float')
        image = (image @ Matrix_rgb888_to_yuv + 0.5).astype('uint8')
        return image

    def _ColorConversion_RGB888_to_YUV422(self, image):
        # rgb888 to yuv444
        image = self._ColorConversion_RGB888_to_YUV444(image)

        # yuv444 to yuv422
        u2 = image[:, 0::2, 1]
        u4 = np.repeat(u2, 2, axis=1)
        v2 = image[:, 1::2, 2]
        v4 = np.repeat(v2, 2, axis=1)
        image[..., 1] = u4
        image[..., 2] = v4
        return image
           
    def _ColorConversion_YUV_to_RGB888(self, image):
        ## fixed
        h, w, c = image.shape
        image_f = image.reshape((h * w, c))
        image_rgb_f = np.zeros(image_f.shape, dtype=np.uint8)

        for i in range(h * w):
            image_y = image_f[i, 0] *1024
            if image_f[i, 1] > 127:
                image_u = -((~(image_f[i, 1] - 1)) & 0xFF)
            else:
                image_u = image_f[i, 1]
            if image_f[i, 2] > 127:
                image_v = -((~(image_f[i, 2] - 1)) & 0xFF)
            else:
                image_v = image_f[i, 2]

            image_r = c00_yuv * image_y + c02_yuv * image_v
            image_g = c10_yuv * image_y + c11_yuv * image_u + c12_yuv * image_v
            image_b = c20_yuv * image_y + c21_yuv * image_u

            image_r = signed_rounding(image_r, format_bit)
            image_g = signed_rounding(image_g, format_bit)
            image_b = signed_rounding(image_b, format_bit)

            image_r = image_r >> format_bit
            image_g = image_g >> format_bit
            image_b = image_b >> format_bit

            image_rgb_f[i, 0] = clip(image_r, 0, 255)
            image_rgb_f[i, 1] = clip(image_g, 0, 255)
            image_rgb_f[i, 2] = clip(image_b, 0, 255)

        image_rgb = image_rgb_f.reshape((h, w, c))
        return image_rgb

    def _ColorConversion_RGB888_to_YCbCr444(self, image):
        ## floating
        image = image.astype('float')
        image = (image @ Matrix_rgb888_to_ycbcr + 0.5).astype('uint8')
        image[:, :, 0] += 16
        image[:, :, 1] += 128
        image[:, :, 2] += 128

        return image

    def _ColorConversion_RGB888_to_YCbCr422(self, image):
        # rgb888 to ycbcr444
        image = self._ColorConversion_RGB888_to_YCbCr444(image)

        # ycbcr444 to ycbcr422
        cb2 = image[:, 0::2, 1]
        cb4 = np.repeat(cb2, 2, axis=1)
        cr2 = image[:, 1::2, 2]
        cr4 = np.repeat(cr2, 2, axis=1)
        image[..., 1] = cb4
        image[..., 2] = cr4
        return image

    def _ColorConversion_YCbCr_to_RGB888(self, image):
        ## floating
        if (self.set['numerical_type'] == 'floating'):
            image = image.astype('float')
            image[:, :, 0] -= 16
            image[:, :, 1] -= 128
            image[:, :, 2] -= 128
            image = ((image @ Matrix_ycbcr_to_rgb888) + 0.5).astype('uint8')
            return image

        ## fixed
        h, w, c = image.shape
        image_f = image.reshape((h * w, c))
        image_rgb_f = np.zeros(image_f.shape, dtype=np.uint8)

        for i in range(h * w):
            image_y = (image_f[i, 0] - 16) * c00_ycbcr
            image_cb = image_f[i, 1] - 128
            image_cr = image_f[i, 2] - 128

            image_r = image_y + c02_ycbcr * image_cr
            image_g = image_y + c11_ycbcr * image_cb + c12_ycbcr * image_cr
            image_b = image_y + c21_ycbcr * image_cb

            image_r = signed_rounding(image_r, format_bit)
            image_g = signed_rounding(image_g, format_bit)
            image_b = signed_rounding(image_b, format_bit)

            image_r = image_r >> format_bit
            image_g = image_g >> format_bit
            image_b = image_b >> format_bit

            image_rgb_f[i, 0] = clip(image_r, 0, 255)
            image_rgb_f[i, 1] = clip(image_g, 0, 255)
            image_rgb_f[i, 2] = clip(image_b, 0, 255)

        image_rgb = image_rgb_f.reshape((h, w, c))
        return image_rgb

    def _ColorConversion_RGB888_to_RGB565(self, image):
        assert (len(image.shape)==3)
        assert (image.shape[2]>=3)
        
        image_rgb565 = np.zeros(image.shape, dtype=np.uint8)
        image_rgb = image.astype('uint8')
        image_rgb565[:, :, 0] = image_rgb[:, :, 0] >> 3
        image_rgb565[:, :, 1] = image_rgb[:, :, 1] >> 2
        image_rgb565[:, :, 2] = image_rgb[:, :, 2] >> 3
        return image_rgb565

    def _ColorConversion_RGB565_to_RGB888(self, image):
        assert (len(image.shape)==3)
        assert (image.shape[2]==3)

        image_rgb = np.zeros(image.shape, dtype=np.uint8)
        image_rgb[:, :, 0] = image[:, :, 0] << 3
        image_rgb[:, :, 1] = image[:, :, 1] << 2
        image_rgb[:, :, 2] = image[:, :, 2] << 3
        return image_rgb

    def _ColorConversion_L_to_RGB888(self, image):
        image_L = image.astype('uint8')
        img = Image.fromarray(image_L).convert('RGB')
        image_data = np.array(img).astype('uint8')
        return image_data

    def _ColorConversion_RGB888_to_L(self, image):
        image_rgb = image.astype('uint8')
        img = Image.fromarray(image_rgb).convert('L')
        image_data = np.array(img).astype('uint8')
        return image_data

    def _ColorConversion_RGBA8888_to_RGB888(self, image):
        assert (len(image.shape)==3)
        assert (image.shape[2]==4)
        return image[:,:,:3]

    def _ColorConversion_RGB888_to_RGBA8888(self, image):
        assert (len(image.shape)==3)
        assert (image.shape[2]==3)
        imageA = np.concatenate((image, np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8) ), axis=2)
        return imageA
