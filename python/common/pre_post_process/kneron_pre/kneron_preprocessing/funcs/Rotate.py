import numpy as np
from .utils import str2bool, str2int

class runner(object):
    def __init__(self, *args, **kwargs):
        self.set = {
            'operator': '',
            "rotate_direction": 0,

        }
        self.update(*args, **kwargs)

    def update(self, *args, **kwargs):
        self.set.update(kwargs)
        self.rotate_direction = str2int(self.set['rotate_direction'])

        # print info
        if str2bool(self.set['b_print']):
            self.print_info()

    def print_info(self):
        print("<rotate>",
            'rotate_direction', self.rotate_direction,)


    def run(self, image_data):
        image_data = self._rotate(image_data)
        return image_data

    def _rotate(self,img):
        if self.rotate_direction == 1 or self.rotate_direction == 2:
            col, row, unit = img.shape
            pInBuf = img.reshape((-1,1))
            pOutBufTemp = np.zeros((col* row* unit))
            for r in range(row):
                for c in range(col):
                    for u in range(unit):
                        if self.rotate_direction == 1:
                            pOutBufTemp[unit * (c * row + (row - r - 1))+u] = pInBuf[unit * (r * col + c)+u]
                        elif self.rotate_direction == 2:
                            pOutBufTemp[unit * (row * (col - c - 1) + r)+u] = pInBuf[unit * (r * col + c)+u]

            img = pOutBufTemp.reshape((col,row,unit))

        return img
