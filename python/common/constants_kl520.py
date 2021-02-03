"""
Constants used  in the examples calling the Python API.
"""
import ctypes

# Application IDs
APP_UNKNOWN = 0
APP_SFID = 2
APP_DME = 3
APP_AGE_GENDER = 4
APP_OD = 5
APP_TINY_YOLO3 = 6
APP_FD_LM = 7
APP_ID_MAX = 8

# Error message values
MSG_APP_BAD_IMAGE = 0x100
MSG_APP_BAD_INDEX = 0x101
MSG_APP_UID_EXIST = 0x102
MSG_APP_UID_DIFF = 0x103
MSG_APP_IDX_FIRST = 0x104
MSG_APP_IDX_MISSING = 0x105
MSG_APP_DB_NO_USER = 0x106
MSG_APP_DB_FAIL = 0x107
MSG_APP_LV_FAIL = 0x108

# Image format flags
IMAGE_FORMAT_SUB128 = 1 << 31
IMAGE_FORMAT_RAW_OUTPUT = 1 << 28
IMAGE_FORMAT_PARALLEL_PROC = 1 << 27
IMAGE_FORMAT_MODEL_AGE_GENDER = 1 << 24
# right shift for 1-bit if 1
IMAGE_FORMAT_RIGHT_SHIFT_ONE_BIT = 1 << 22
IMAGE_FORMAT_SYMMETRIC_PADDING = 1 << 21
IMAGE_FORMAT_CHANGE_ASPECT_RATIO = 1 << 20
IMAGE_FORMAT_BYPASS_PRE = 1 << 19
IMAGE_FORMAT_BYPASS_NPU_OP = 1 << 18
IMAGE_FORMAT_BYPASS_CPU_OP = 1 << 17
IMAGE_FORMAT_BYPASS_POST = 1 << 16

NPU_FORMAT_RGBA8888 = 0x00
NPU_FORMAT_NIR = 0x20
# Support YCBCR (YUV)
NPU_FORMAT_YCBCR422 = 0x30
NPU_FORMAT_YCBCR444 = 0x50
NPU_FORMAT_RGB565 = 0x60

# Used for OD examples
OD_MIN_PERSON_SCORE = 0.2
OD_MIN_OTHER_CLASS_SCORE = 0.4
OD_BBOX_RATIO = 0.93
OD_CLASSES = ['background', 'bicycle', 'bus', 'car', 'cat', 'dog', 'motorbike', 'person']

DME_OBJECT_MAX = 80

MAX_PARAMS_LEN = 40

# Structures used for the C shared library.
class KDPDMEConfig(ctypes.Structure):
    """Image configuration structure"""
    _fields_ = [("model_id", ctypes.c_int),       # int32_t
                ("output_num", ctypes.c_int),     # int32_t
                ("image_col", ctypes.c_int),      # int32_t
                ("image_row", ctypes.c_int),      # int32_t
                ("image_ch", ctypes.c_int),       # int32_t
                ("image_format", ctypes.c_uint),  # uint32_t
                ("crop_box", ctypes.c_int * 4),   # y1,y2,x1,x2, for future use
                ("pad_value", ctypes.c_int * 4),  # for future use
                ("extra_param", ctypes.c_float * MAX_PARAMS_LEN)] # extra parameters, such as threshold

    def __init__(self, model_id, output_num, col, row, ch, image_format, crop_box, pad_value, extra_param):
        self.model_id = model_id
        self.output_num = output_num
        self.image_col = col
        self.image_row = row
        self.image_ch = ch
        self.image_format = image_format
        self.crop_box = ctypes.cast(crop_box, ctypes.POINTER(ctypes.c_int * 4)).contents
        self.pad_value = ctypes.cast(pad_value, ctypes.POINTER(ctypes.c_int * 4)).contents
        self.extra_param = ctypes.cast(extra_param, ctypes.POINTER(ctypes.c_float * MAX_PARAMS_LEN)).contents

    def __repr__(self):
        return "id: {}, output_num: {}, dims: ({}, {}, {}), format: {}".format(
            self.model_id, self.output_num, self.image_ch, self.image_row, self.image_col, self.image_format)

class KDPISIConfig(ctypes.Structure):
    """Image configuration structure"""
    _fields_ = [("app_id", ctypes.c_uint),           # uint32_t
                ("res_buf_size", ctypes.c_uint),     # uint32_t
                ("image_col", ctypes.c_ushort),      # uint16_t
                ("image_row", ctypes.c_ushort),      # uint16_t
                ("image_format", ctypes.c_uint),     # uint32_t
                ("crop_box", ctypes.c_int * 4),      # y1,y2,x1,x2, for future use
                ("pad_value", ctypes.c_int * 4),     # for future use
                ("extra_param", ctypes.c_float * MAX_PARAMS_LEN)] # extra parameters, such as threshold

    def __init__(self, app_id, res_buf_size, col, row, image_format, crop_box, pad_value, extra_param):
        self.app_id = app_id
        self.res_buf_size = res_buf_size
        self.image_col = col
        self.image_row = row
        self.image_format = image_format
        self.crop_box = ctypes.cast(crop_box, ctypes.POINTER(ctypes.c_int * 4)).contents
        self.pad_value = ctypes.cast(pad_value, ctypes.POINTER(ctypes.c_int * 4)).contents
        self.extra_param = ctypes.cast(extra_param, ctypes.POINTER(ctypes.c_float * MAX_PARAMS_LEN)).contents

    def __repr__(self):
        return "app_id: {}, res_buf_size: {},  dims: ({}, {}), format: {}".format(
            self.app_id, self.res_buf_size, self.image_row, self.image_col, self.image_format)

class BoundingBox(ctypes.Structure):
    _fields_ = [("x1", ctypes.c_float),       # float
                ("y1", ctypes.c_float),       # float
                ("x2", ctypes.c_float),       # float
                ("y2", ctypes.c_float),       # float
                ("score", ctypes.c_float),    # float
                ("class_num", ctypes.c_int)]  # int32_t

class AgeGenderResult(ctypes.Structure):
    _fields_ = [("age", ctypes.c_uint),     # uint32_t
                ("ismale", ctypes.c_uint)]  # uint32_t

class FDAgeGenderRes(ctypes.Structure):
    _fields_ = [("fd_res", BoundingBox),
                ("ag_res", AgeGenderResult)]

class FDAgeGenderS(ctypes.Structure):
    _fields_ = [("count", ctypes.c_uint),
                ("boxes", FDAgeGenderRes * 0)]

class ObjectDetectionRes(ctypes.Structure):
    _fields_ = [("class_count", ctypes.c_uint), # uint32_t
                ("box_count", ctypes.c_uint),   # boxes of all classes
                ("boxes", BoundingBox * 0)]     # box array

class OutputNodeParams(ctypes.Structure):
    _fields_ = [("height", ctypes.c_int),     # int32_t
                ("channel", ctypes.c_int),    # int32_t
                ("width", ctypes.c_int),      # int32_t
                ("radix", ctypes.c_int),      # int32_t
                ("scale", ctypes.c_float)]    # float

class RawFixpointData(ctypes.Structure):
    _fields_ = [("output_num", ctypes.c_uint),             # uint32_t
                ("out_node_params", OutputNodeParams * 0), # out node params array
                ("fp_data", ctypes.c_char * 0)]            # fixed-point data array

class FDResult(ctypes.Structure):
    _fields_ = [("x", ctypes.c_short),       # int16_t
                ("y", ctypes.c_short),       # int16_t
                ("w", ctypes.c_short),       # int16_t
                ("h", ctypes.c_short)]       # int16_t

class LMPoint(ctypes.Structure):
    _fields_ = [("x", ctypes.c_short),       # int16_t
                ("y", ctypes.c_short)]       # int16_t

class LMResult(ctypes.Structure):
    """Landmark result structure in FDLM Result for FDR application"""
    _fields_ = [("marks", LMPoint * 5)]      # LMPoint array

class FDLMRes(ctypes.Structure):
    _fields_ = [("fd_res", FDResult),
                ("lm_res", LMResult)]
class LandmarkPoint(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int),       # int32_t
                ("y", ctypes.c_int)]       # int32_t

class LandmarkResult(ctypes.Structure):
    """Landmark result structure in LM Result for other applications"""
    _fields_ = [("marks", LandmarkPoint * 5),     # LMPoint array
                ("score", ctypes.c_float),        # float
                ("blur", ctypes.c_float)]         # float

class NefModelMetaData(ctypes.Structure):
    _fields_ = [("platform", ctypes.c_char * 32),     # char array: usb dongle, 96 board, etc.
                ("target", ctypes.c_uint),            # uint32_t: 0 for KL520, 1 for KL720, etc.
                ("crc", ctypes.c_uint),               # uint32_t: crc value for all_models data
                ("kn_num", ctypes.c_uint),            # uint32_t
                ("enc_type", ctypes.c_uint),          # uint32_t: encrypt type
                ("tc_ver", ctypes.c_char * 32),       # char array: toolchain version
                ("compiler_ver", ctypes.c_char * 32)] # char array: compiler version