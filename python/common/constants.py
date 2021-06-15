"""
Constants used for various examples calling the Python host API.
"""
import ctypes
import enum

## Structures used for C shared library
class KDPClass(ctypes.Structure):
    """Basic class for KDP structs."""
    def struct_size(self):
        # This may be different from C version, depending on how arrays are initialized in C.
        return ctypes.sizeof(type(self))

    def __eq__(self, other):
        for field in self._fields_:
            if getattr(self, field[0]) != getattr(other, field[0]):
                return False
        return True

    def __ne__(self, other):
        for field in self._fields_:
            if getattr(self, field[0]) != getattr(other, field[0]):
                return True
        return False

## from common/base.h
def bit(num):
    return 1 << num

# Application IDs, from common/kapp_id.h
@enum.unique
class AppID(enum.IntEnum):
    """Enumeration of application IDs."""
    # available device:               KL520       KL720
    APP_UNKNOWN = 0
    APP_RESERVE = enum.auto()       #   v
    APP_SFID = enum.auto()          #   v
    APP_DME = enum.auto()           #   v           v
    APP_AGE_GENDER = enum.auto()    #   v
    APP_OD = enum.auto()            #   v
    APP_TINY_YOLO3 = enum.auto()    #   v
    APP_FD_LM = enum.auto()         #   v
    APP_PD = enum.auto()            #   v
    APP_CENTER_APP = enum.auto()    #               v
    APP_2PASS_OD = enum.auto()      #               v
    APP_JPEG_ENC = enum.auto()      #               v
    APP_JPEG_DEC = enum.auto()      #               v
    APP_YAWNING = enum.auto()       #   v
    APP_HICO = enum.auto()          #               v
    APP_TD = enum.auto()            #   v
    APP_PDC = enum.auto()           #               v
    APP_CAR_PLATE = enum.auto()     #               v
    APP_ID_MAX = enum.auto()        # must be last entry

# Communication errors, from common/com_err.h
@enum.unique
class ComErr(enum.IntEnum):
    """Enumeration of communication errors."""
    SUCCESS = 0
    UNKNOWN_ERR = -1
    MSG_APP_DB_NO_MATCH = 10
    MSG_APP_NOT_LOADED = 100
    MSG_SYSTEM_OVERHEATING_ERROR = 119      # System is overheating
    MSG_APP_BAD_MODE = 245                  # bad application mode
    MSG_DB_ADD_FM_FAIL = enum.auto()        # add DB uid_indx failed
    MSG_DB_DEL_FM_FAIL = enum.auto()        # delete DB uid_indx failed
    MSG_DB_BAD_PARAM = enum.auto()          # database action/format error
    MSG_SFID_OUT_FAIL = enum.auto()         # data upload failed
    MSG_SFID_NO_MEM = enum.auto()           # memory allocation failed
    MSG_AUTH_FAIL = enum.auto()             # authentication failed
    MSG_FLASH_FAIL = enum.auto()            # flash programming failed (bad sector?)
    MSG_DATA_ERROR = enum.auto()            # data error (I/O)
    MSG_SFID_USR_ZERO = enum.auto()         # user id zero error
    MSG_CONFIG_ERROR = 255                  # no appropriate Start issued previously
    MSG_APP_BAD_IMAGE = 256
    MSG_APP_BAD_INDEX = enum.auto()
    MSG_APP_UID_EXIST = enum.auto()
    MSG_APP_UID_DIFF = enum.auto()
    MSG_APP_IDX_FIRST = enum.auto()
    MSG_APP_IDX_MISSING = enum.auto()
    MSG_APP_DB_NO_USER = enum.auto()
    MSG_APP_DB_FAIL = enum.auto()
    MSG_APP_LV_FAIL = enum.auto()
    MSG_DB_QUERY_FM_FAIL = enum.auto()      # query DB fm data failed

# Model IDs, from common/model_type.h
class ModelType(enum.IntEnum):
    """Enumeration of model types."""
    INVALID_ID = 0
    INVALID_TYPE = 0
    KNERON_FD_SMALLBOX_200_200_3 = 1
    KNERON_FD_ANCHOR_200_200_3 = 2
    KNERON_FD_MBSSD_200_200_3 = 3
    AVERAGE_POOLING = 4, # use with FD smallbox and don't use anymore
    KNERON_LM_5PTS_ONET_56_56_3 = 5
    KNERON_LM_68PTS_dlib_112_112_3 = 6
    KNERON_LM_150PTS = 7
    KNERON_FR_RES50_112_112_3 = 8
    KNERON_FR_RES34 = 9
    KNERON_FR_VGG10 = 10
    KNERON_TINY_YOLO_PERSON_416_416_3 = 11
    KNERON_3D_LIVENESS = 12 # has two inputs: depth and RGB
    KNERON_GESTURE_RETINANET_320_320_3 = 13
    TINY_YOLO_VOC_224_224_3 = 14
    IMAGENET_CLASSIFICATION_RES50_224_224_3 = 15
    IMAGENET_CLASSIFICATION_RES34_224_224_3 = 16
    IMAGENET_CLASSIFICATION_INCEPTION_V3_224_224_3  = 17
    IMAGENET_CLASSIFICATION_MOBILENET_V2_224_224_3  = 18
    TINY_YOLO_V3_224_224_3 = 19
    KNERON_2D_LIVENESS_224_224_3 = 20 # oldest rgb liveness model and don't use anymore
    KNERON_FD_RETINANET_256_256_3 = 21
    KNERON_PERSON_MOBILENETSSD_224_224_3 = 22
    KNERON_AGE_GENDER = 23 # oldest age gender model and don't use anymore
    KNERON_LM_5PTS_BLUR_ONET_48_48_3 = 24
    KNERON_2D_LIVENESS_V3_FACEBAGNET_224_224_3 = 25
    KNERON_AGE_GENDER_V2_RES18_128_128_3 = 26
    KNERON_OD_MBSSD = 27 # HW model and don't know input size
    KNERON_PD_MBSSD = 28 # HW model and don't know which version and input size
    KNERON_FR_MASK_RES50_112_112_3 = 29
    KNERON_NIR_LIVENESS_RES18_112_112_3 = 30
    KNERON_FR_MASK_RES101_112_112_3 = 31
    KNERON_FD_MASK_MBSSD_200_200_3 = 32
    TINY_YOLO_V3_416_416_3 = 33
    TINY_YOLO_V3_608_608_3 = 34

    # Category Face related 40~200
    KNERON_CAT_FACE = 40
    KNERON_FACE_QAULITY_ONET_56_56_1 = KNERON_CAT_FACE
    KNERON_FUSE_LIVENESS = KNERON_CAT_FACE + 1 # don't know the model backbone and 
                                               # input size of fuse liveness model
    KNERON_EYELID_DETECTION_ONET_48_48_3 = KNERON_CAT_FACE + 2
    KNERON_YAWN_DETECTION_PFLD_112_112_3 = KNERON_CAT_FACE + 3
    KNERON_DBFACE_MBNET_V2_480_864_3 = KNERON_CAT_FACE + 4
    KNERON_FILTER = KNERON_CAT_FACE + 5 # No model inference, just pre and post-process
    KNERON_ALIGNMENT = KNERON_CAT_FACE + 6 # No model inference, just preprocess
    KNERON_FACE_EXPRESSION_112_112_3 = KNERON_CAT_FACE + 7
    KNERON_RBG_OCCLUSION_RES18_112_112_3 = KNERON_CAT_FACE + 8
    KNERON_LM2BBOX = KNERON_CAT_FACE + 9 # No model inference, just post-process
    KNERON_PUPIL_ONET_48_48_3 = KNERON_CAT_FACE + 10
    KNERON_NIR_OCCLUSION_RES18_112_112_3 = KNERON_CAT_FACE + 11
    KNERON_HEAD_SHOULDER_MBNET_V2_112_112_3 = KNERON_CAT_FACE + 12
    KNERON_RGB_LIVENESS_RES18_112_112_3 = KNERON_CAT_FACE + 13
    KNERON_MOUTH_LM_v1_56_56_1 = KNERON_CAT_FACE + 14 # nose, upper lip middle, chin,
                                                      # two sides of faces
    KNERON_MOUTH_LM_v2_56_56_1 = KNERON_CAT_FACE + 15 # nose, upper/lower lip middle,
                                                      # two sides of faces
    KNERON_PUPIL_ONET_48_48_1 = KNERON_CAT_FACE + 16
    KNERON_RGB_LIVENESS_MBV2_112_112_3 = KNERON_CAT_FACE + 17
    KNERON_FACESEG_DLA34_128_128_3 = KNERON_CAT_FACE + 18
    KNERON_OCC_CLS = KNERON_CAT_FACE + 19 # no model inference, just post-process
    KNERON_LMSEG_FUSE = KNERON_CAT_FACE + 20 # no model inference, just post-process
    KNERON_AGEGROUP_RES18_128_128_3 = KNERON_CAT_FACE + 21
    KNERON_FR_kface_112_112_3 = KNERON_CAT_FACE + 22
    KNERON_FD_ROTATE_MBSSD_200_200_3 = KNERON_CAT_FACE + 23
    KNERON_LM_5PTSROTATE_ONET_56_56_3 = KNERON_CAT_FACE + 24
    KNERON_FUSE_LIVENESS_850MM = KNERON_CAT_FACE + 25
    KNERON_FUSE_LIVENESS_940MM = KNERON_CAT_FACE + 26
    KNERON_FACE_QAULITY_ONET_112_112_3 = KNERON_CAT_FACE + 27
    KNERON_FACE_POSE_ONET_56_56_3 = KNERON_CAT_FACE + 28
    KNERON_FUSE_LIVENESS_850MM_RES18_112_112_3 = KNERON_CAT_FACE + 29 # Gen's resnet+agg fuse model
    KNERON_OCCCLASSIFER_112_112_3 = KNERON_CAT_FACE + 30
    KNERON_FACE_ROTATE_POSE_ONET_56_56_3 = KNERON_CAT_FACE + 31
    KNERON_NIR_LIVENESS_ROT_RES18_112_112_3 = KNERON_CAT_FACE + 32
    KNERON_LM_5PTS_ONETPLUS_56_56_3 = KNERON_CAT_FACE + 33
    KNERON_FUSE_LIVENESS_940MM_18_RES18_112_112_3 = KNERON_CAT_FACE + 34
    KNERON_DBFACE_MBNET_V2_256_352_3 = KNERON_CAT_FACE + 35

    # Category Object Detection related 200~300
    KNERON_OB_DETECT = 200
    KNERON_OBJECTDETECTION_CENTERNET_512_512_3 = KNERON_OB_DETECT
    KNERON_OBJECTDETECTION_FCOS_416_416_3 = KNERON_OB_DETECT + 1
    KNERON_PD_MBNET_V2_480_864_3 = KNERON_OB_DETECT + 2 # 16:9 aspect ratio
    KNERON_CAR_DETECTION_MBSSD_224_416_3 = KNERON_OB_DETECT + 3
    KNERON_PD_CROP_MBSSD_304_304_3 = KNERON_OB_DETECT + 4
    YOLO_V3_416_416_3 = KNERON_OB_DETECT + 5
    YOLO_V4_416_416_3 = KNERON_OB_DETECT + 6
    KNERON_CAR_DETECTION_YOLO_V5_352_640_3 = KNERON_OB_DETECT + 7
    KNERON_LICENSE_DETECT_WPOD_208_416_3 = KNERON_OB_DETECT + 8
    KNERON_2D_UPPERBODY_KEYPOINT_RES18_384_288_3 = KNERON_OB_DETECT + 9
    YOLO_V3_608_608_3 = KNERON_OB_DETECT + 10
    KNERON_YOLOV5S_640_640_3 = KNERON_OB_DETECT + 11
    KNERON_YOLOV5S_480_256_3 = KNERON_OB_DETECT + 12
    KNERON_SITTINGPOSTURE_RESNET34_288_384_3 = KNERON_OB_DETECT + 13
    KNERON_PERSONDETECTION_FCOS_416_416_3 = KNERON_OB_DETECT + 14
    KNERON_YOLOV5M_640_640_3 = KNERON_OB_DETECT + 15
    KNERON_YOLOV5S6_480_256_3  = KNERON_OB_DETECT + 16
    KNERON_PERSONDETECTION_FCOS_384_288_3 = KNERON_OB_DETECT + 17
    KNERON_PERSONDETECTION_FCOS_720_416_3 = KNERON_OB_DETECT + 18
    KNERON_PERSONDETECTION_dbface_864_480_3 = KNERON_OB_DETECT + 19
    KNERON_PERSONDETECTION_YOLOV5s_480_256_3 = KNERON_OB_DETECT + 20
    KNERON_PERSONCLASSIFIER_MB_56_32_3 = KNERON_OB_DETECT + 21
    KNERON_PERSONREID_RESNET_42_82_3 = KNERON_OB_DETECT + 22
    KNERON_PERSONDETECTION_YOLOV5s_928_512_3 = KNERON_OB_DETECT + 23
    KNERON_UPKPTS_RSN_256_192_3 = KNERON_OB_DETECT + 24
    KNERON_PERSONDETECTION_YOLOV5sParklot_480_256_3 = KNERON_OB_DETECT + 25
    KNERON_CAR_DETECTION_MBSSD_304_544_3 = KNERON_OB_DETECT + 26
    KNERON_KPTSCLASSIFIER_3_11_1 = KNERON_OB_DETECT + 27
    KNERON_YOLOV5S3_480_256_3 = KNERON_OB_DETECT + 28

    # Category OCR related 300~400
    KNERON_OCR = 300
    KNERON_LICENSE_OCR_MBNET_64_160_3 = KNERON_OCR
    KNERON_WATERMETER_OCR_MBNET = KNERON_OCR + 1 # unknown
    KNERON_LICENSE_OCR_MBNETv2_64_160_3 = KNERON_OCR + 2

    # Category SDK test related
    KNERON_CAT_SDK_TEST = 1000
    KNERON_SDK_FD = KNERON_CAT_SDK_TEST
    KNERON_SDK_LM = KNERON_CAT_SDK_TEST + 1
    KNERON_SDK_FR = KNERON_CAT_SDK_TEST + 2

    # Category Customer modelS, 0x8000 = 32768
    CUSTOMER = 32768
    CUSTOMER_MODEL_1 = CUSTOMER
    CUSTOMER_MODEL_2 = CUSTOMER + 1
    CUSTOMER_MODEL_3 = CUSTOMER + 2
    CUSTOMER_MODEL_4 = CUSTOMER + 3
    CUSTOMER_MODEL_5 = CUSTOMER + 4
    CUSTOMER_MODEL_6 = CUSTOMER + 5
    CUSTOMER_MODEL_7 = CUSTOMER + 6
    CUSTOMER_MODEL_8 = CUSTOMER + 7
    CUSTOMER_MODEL_9 = CUSTOMER + 8

# from common/common.h
FID_THRESHOLD = 0.475 # default comparision threshold for 1st DB

class SFIDConfigType(enum.IntEnum): # TODO check if used
    """Thresholds for matching face recognition result."""
    SFID_FR_UNMASKED_THRESHOLD = 0          # threshold used to match unmasked face recognition
    SFID_FR_MASKED_THRESHOLD = enum.auto()  # threshold used to match masked face recognition

# Postprocessing result info, from common/model_result.h, common/model_res.h
LAND_MARK_POINTS = 5
FR_FEAT_SIZE = 256
LV_R_SIZE = 1
DUAL_LAND_MARK_POINTS = 10
DME_OBJECT_MAX = 80
IMAGENET_TOP_MAX = 5
LAND_MARK_MOUTH_POINTS = 4

FD_RES_LENGTH = 2 * 5

YOLO_DETECTION_MAX = 80
MAX_CRC = 4
FACE_DETECTION_XYWH = 4
FR_FEATURE_MAP_SIZE = 512

class BoundingBox(KDPClass):
    """Bounding box result."""
    _fields_ = [("x1", ctypes.c_float),         # top left x coordinate
                ("y1", ctypes.c_float),         # top left y coordinate
                ("x2", ctypes.c_float),         # bottom right x coordinate
                ("y2", ctypes.c_float),         # bottom right y coordinate
                ("score", ctypes.c_float),      # probability score
                ("class_num", ctypes.c_int32)]  # class number with highest probability

    def __init__(self, x1=0, y1=0, x2=0, y2=0, score=0, class_num=0):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.score = score
        self.class_num = class_num

    def __repr__(self):
        return ("------------------------------ Bounding Box -------------------------------\n"
                f"Box (x1, y1, x2, y2): ({self.x1}, {self.y1}, {self.x2}, {self.y2})\nScore: "
                f"{self.score}\nClass number: {self.class_num}\n")

class YoloResult(KDPClass):
    """Result of all bounding boxes in image."""
    # struct_size will differ from C because of 'boxes'
    _fields_ = [("class_count", ctypes.c_uint32),               # total class count
                ("box_count", ctypes.c_uint32),                 # boxes of all classes
                ("boxes", BoundingBox * YOLO_DETECTION_MAX)]    # list of bounding boxes

    def __init__(self, class_count=0, box_count=0, boxes=[BoundingBox()]):
        self.class_count = class_count
        self.box_count = box_count
        self.boxes = (BoundingBox * YOLO_DETECTION_MAX)(*boxes)

    def __repr__(self):
        box_repr = ""
        for index, box in enumerate(self.boxes[:self.box_count]):
            box_repr += f"Box #{index}:\n{box}\n"
        return ("------------------------------- Yolo Result -------------------------------\n"
                f"Class count: {self.class_count}\nBox_count: {self.box_count}\n{box_repr}\n")

class AgeGenderResult(KDPClass):
    _fields_ = [("age", ctypes.c_uint32),         # age
                ("ismale", ctypes.c_uint32)]      # gender

    def __init__(self, age=0, ismale=0):
        self.age = age
        self.ismale = ismale

    def __repr__(self):
        gender = "male" if self.ismale else "female"
        return ("---------------------------- Age Gender Result ----------------------------\n"
                f"Age: {self.age}\nGender: {gender}\n")

class ImageNetResult(KDPClass):
    """Image net result."""
    _fields_ = [("index", ctypes.c_int32),      # class index
                ("score", ctypes.c_float)]      # probability score

    def __init__(self, index=0, score=0):
        self.index = index
        self.score = score

    def __repr__(self):
        return ("----------------------------- ImageNet Result -----------------------------\n"
                f"Class index: {self.index}\nScore: {self.score}\n")

class LandmarkPoint(KDPClass):
    """Landmark point."""
    _fields_ = [("x", ctypes.c_uint32),     # x coordinate
                ("y", ctypes.c_uint32)]     # y coordinate

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Landmark point (x, y): ({self.x}, {self.y})"

class LandmarkResult(KDPClass):
    """Landmark result structure in FDLM Result for FDR application"""
    _fields_ = [("marks", LandmarkPoint * LAND_MARK_POINTS),  # list of landmark points
                ("score", ctypes.c_float),
                ("blur", ctypes.c_float),
                ("class_num", ctypes.c_int32)]

    def __init__(self, marks=[LandmarkPoint()], score=0, blur=0, class_num=0):
        self.marks = (LandmarkPoint * LAND_MARK_POINTS)(*marks)
        self.score = score
        self.blur = blur
        self.class_num = class_num

    def __repr__(self):
        lm = ""
        for index, point in enumerate(self.marks):
            lm += f"#{index} {point}\n"
        return ("----------------------------- Landmark Result -----------------------------\n"
                f"Landmarks:\n{lm}\nScore: {self.score}\nBlur: {self.blur}\nClass number: "
                f"{self.class_num}")

class FRResult(KDPClass):
    """Face recognition feature map."""
    _fields_ = [("feature_map", ctypes.c_float * FR_FEAT_SIZE),     # feature map in float
                ("feature_map_fixed", ctypes.c_int8 * FR_FEAT_SIZE)]# feature map in int

    def __init__(self, feature_map=[0], feature_map_fixed=[0]):
        self.feature_map = (ctypes.c_float * FR_FEAT_SIZE)(*feature_map)
        self.feature_map_fixed = (ctypes.c_int8 * FR_FEAT_SIZE)(*feature_map_fixed)

    def __repr__(self):
        fr = ""
        for index, (fr_float, fr_int) in enumerate(
            zip(self.feature_map[:10], self.feature_map_fixed[:10])):
            fr += f"FR[{index}] (float, fixed): {fr_float}, {fr_int}\n"
        return ("------------------------- Face Recognition Result -------------------------\n"
                f"Only showing first 10 results.\n{fr}")

class FDAgeGenderRes(KDPClass):
    _fields_ = [("fd_res", BoundingBox),        # FD result
                ("ag_res", AgeGenderResult)]    # Age/gender result

    def __init__(self, fd_res=BoundingBox(), ag_res=AgeGenderResult()):
        self.fd_res = fd_res
        self.ag_res = ag_res

    def __repr__(self):
        return ("-------------------- Face Detection, Age Gender Result --------------------\n"
                f"{self.fd_res}\n{self.ag_res}\n")

class FDAgeGenderS(KDPClass):
    _fields_ = [("count", ctypes.c_uint32),                 # result count
                ("boxes", FDAgeGenderRes * DME_OBJECT_MAX)] # list of FD/AgeGender results

    def __init__(self, count=0, boxes=[FDAgeGenderRes()]):
        self.count = count
        self.boxes = (FDAgeGenderRes * DME_OBJECT_MAX)(*boxes)

    def __repr__(self):
        boxes = "\n"
        for index, box in enumerate(self.boxes[:self.count]):
            boxes += f"FD Age Gender result #{index}\n{box}\n"
        return ("------------------- Face Detection, Age Gender Results --------------------\n"
                f"Result count: {self.count}\n{boxes}\n")

class OutputNodeParams720(ctypes.Structure):
    _fields_ = [("start_offset", ctypes.c_uint),
                ("buf_len", ctypes.c_uint),
                ("node_id", ctypes.c_uint),
                ("supernum", ctypes.c_uint),
                ("data_format", ctypes.c_uint),
                ("row_start", ctypes.c_uint),
                ("col_start", ctypes.c_uint),
                ("ch_start", ctypes.c_uint),
                ("height", ctypes.c_int),        # int32_t
                ("width", ctypes.c_int),         # int32_t
                ("channel", ctypes.c_int),       # int32_t
                ("output_index", ctypes.c_uint), # uint32_t
                ("radix", ctypes.c_int),         # int32_t
                ("scale", ctypes.c_float)]       # float

MAX_NODE_RAW_HEADER = 40

class RawFixpointData720(ctypes.Structure):
    _fields_ = [("total_raw_len", ctypes.c_uint),  # uint32_t
                ("total_nodes", ctypes.c_int),     # int32_t
                ("out_node_params", OutputNodeParams720 * MAX_NODE_RAW_HEADER),  # out node params array
                ("fp_data", ctypes.c_char * 0)]    # fixed-point data array

# from constants_kl520 merge
class ObjectDetectionRes(KDPClass):
    _fields_ = [("class_count", ctypes.c_uint32),   # uint32_t
                ("box_count", ctypes.c_uint),       # boxes of all classes
                ("boxes", BoundingBox * 0)]         # box array

class OutputNodeParams(KDPClass):
    _fields_ = [("height", ctypes.c_int),     # int32_t
                ("channel", ctypes.c_int),    # int32_t
                ("width", ctypes.c_int),      # int32_t
                ("radix", ctypes.c_int),      # int32_t
                ("scale", ctypes.c_float)]    # float

class RawFixpointData(KDPClass):
    _fields_ = [("output_num", ctypes.c_uint),             # uint32_t
                ("out_node_params", OutputNodeParams * 0), # out node params array
                ("fp_data", ctypes.c_char * 0)]            # fixed-point data array

class FDResult(KDPClass):
    """Face detection result."""
    _fields_ = [("x", ctypes.c_short),
                ("y", ctypes.c_short),
                ("w", ctypes.c_short),
                ("h", ctypes.c_short)]

    def __init__(self, x=0, y=0, w=0, h=0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def __repr__(self):
        return ("-------------------------------- FD Result --------------------------------\n"
                f"Box (x, y, w, h): ({self.x}, {self.y}, {self.w}, {self.h})\n")

class LMPoint(KDPClass):
    """520 landmark point."""
    _fields_ = [("x", ctypes.c_short),       # x coordinate
                ("y", ctypes.c_short)]       # y coordinate

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Landmark point (x, y): ({self.x}, {self.y})"

class LMResult(KDPClass):
    """520 landmark result structure in FDLM Result for FDR application."""
    _fields_ = [("marks", LMPoint * 5)]      # LMPoint array

    def __init__(self, marks=[LMPoint()]):
        self.marks = (LMPoint * 5)(*marks)

    def __repr__(self):
        lm = ""
        for index, point in enumerate(self.marks):
            lm += f"#{index} {point}\n"
        return ("----------------------------- Landmark Result -----------------------------\n"
                f"Landmarks:\n{lm}\n")

class FDLMRes(KDPClass):
    """Face detection and landmark results."""
    _fields_ = [("fd_res", FDResult),
                ("lm_res", LMResult)]

    def __init__(self, fd_res=FDResult(), lm_res=LMResult()):
        self.fd_res = fd_res
        self.lm_res = lm_res

    def __repr__(self):
        return ("------------------------------ FD+LM Result -------------------------------\n"
                f"{self.fd_res}\n{self.lm_res}\n")

# from common/ipc.h
IMAGE_FORMAT_SUB128 = bit(31)
IMAGE_FORMAT_ROT_MASK = bit(30) | bit(29)
IMAGE_FORMAT_ROT_SHIFT = 29
IMAGE_FORMAT_ROT_CLOCKWISE = 0x01
IMAGE_FORMAT_ROT_COUNTER_CLOCKWISE = 0x02

IMAGE_FORMAT_RAW_OUTPUT = bit(28)
IMAGE_FORMAT_PARALLEL_PROC = bit(27)

IMAGE_FORMAT_MODEL_AGE_GENDER = bit(24)

IMAGE_FORMAT_RIGHT_SHIFT_ONE_BIT = bit(22)
IMAGE_FORMAT_SYMMETRIC_PADDING = bit(21)
IMAGE_FORMAT_PAD_SHIFT = 21

IMAGE_FORMAT_CHANGE_ASPECT_RATIO = bit(20)

IMAGE_FORMAT_BYPASS_PRE = bit(19)
IMAGE_FORMAT_BYPASS_NPU_OP = bit(18)
IMAGE_FORMAT_BYPASS_CPU_OP = bit(17)
IMAGE_FORMAT_BYPASS_POST = bit(16)

IMAGE_FORMAT_NPU = 0x00FF
NPU_FORMAT_RGBA8888 = 0x00
NPU_FORMAT_NIR = 0x20
NPU_FORMAT_YCBCR422 = 0x30
NPU_FORMAT_YCBCR444 = 0x50
NPU_FORMAT_RGB565 = 0x60

NPU_FORMAT_YCBCR422_CRY1CBY0 = 0x30
NPU_FORMAT_YCBCR422_CBY1CRY0 = 0x31
NPU_FORMAT_YCBCR422_Y1CRY0CB = 0x32
NPU_FORMAT_YCBCR422_Y1CBY0CR = 0x33
NPU_FORMAT_YCBCR422_CRY0CBY1 = 0x34
NPU_FORMAT_YCBCR422_CBY0CRY1 = 0x35
NPU_FORMAT_YCBCR422_Y0CRY1CB = 0x36
NPU_FORMAT_YCBCR422_Y0CBY1CR = 0x37

MAX_PARAMS_LEN = 40

class KDPImgDesc(KDPClass):
    _fields_ = [("image_col", ctypes.c_int32),      # column size
                ("image_row", ctypes.c_int32),      # row size
                ("image_ch", ctypes.c_int32),       # channel size
                ("image_format", ctypes.c_uint32)]  # image format

    def __init__(self, image_col=0, image_row=0, image_ch=0, image_format=0):
        self.image_col = image_col
        self.image_row = image_row
        self.image_ch = image_ch
        self.image_format = image_format

    def __repr__(self):
        return ("-------------------------- KDP Image Description --------------------------\n"
                f"Image dims (c, w, h): ({self.image_ch}, {self.image_col}, {self.image_row})\n"
                f"Image format: {hex(self.image_format)}\n")

class KDPCropBox(KDPClass):
    """Crop box."""
    _fields_ = [("top", ctypes.c_int32),    # y coordinate of top of box
                ("bottom", ctypes.c_int32), # y coordinate of bottom of box
                ("left", ctypes.c_int32),   # x coordinate of left of box
                ("right", ctypes.c_int32)]  # x coordinate of right of box

    def __init__(self, top=0, bottom=0, left=0, right=0):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

    def __repr__(self):
        return f"Crop Box (y1, y2, x1, x2): ({self.top}, {self.bottom}, {self.left}, {self.right})"

class KDPPadValue(ctypes.Structure):
    """Padding."""
    _fields_ = [("pad_top", ctypes.c_int32),    # padding for top of image
                ("pad_bottom", ctypes.c_int32), # padding for bottom of image
                ("pad_left", ctypes.c_int32),   # padding for left of image
                ("pad_right", ctypes.c_int32)]  # padding for right of image

    def __init__(self, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0):
        self.pad_top = pad_top
        self.pad_bottom = pad_bottom
        self.pad_left = pad_left
        self.pad_right = pad_right

    def __repr__(self):
        return (f"Padding (top, bottom, left, right): ({self.pad_top}, "
                f"{self.pad_bottom}, {self.pad_left}, {self.pad_right})")


## from kdp_host.h
KDP_UART_DEV = 0        # identification of using UART I/F
KDP_USB_DEV = 1         # identification of using USB I/F

IMG_FORMAT_RGBA8888 = 0x80000000  # image format: RGBA8888
IMG_FORMAT_RAW8 = 0x80000020      # image format: RAW8
IMG_FORMAT_YCbCr422 = 0x80000037  # image format: YCbCr422 [low byte]Y0CbY1CrY2CbY3Cr...[high byte]
IMG_FORMAT_RGB565 = 0x80000060    # image_format: RGB565

DEF_FR_THRESH = FID_THRESHOLD       # FDR app only, default face recognition threshold
DEF_RGB_FR_THRESH = DEF_FR_THRESH   # FDR app only, default face recognition for RGB source
DEF_NIR_FR_THRESH = DEF_FR_THRESH   # FDR app only, default face recognition for NIR source

## ISI image header flags
FLAGS_PARAMS_HAS_CROP = bit(0)  # image_params[0/1/2/3]: cropping top/bottom/left/right
FLAGS_PARAMS_HAS_PAD = bit(1)   # image_params[n..n+3]: padding top/bottom/left/right
# length of post processing parameters
FLAGS_PARAMS_HAS_POST_PROC_PARAMS = (bit(2) | bit(3) | bit(4) | bit(5) | bit(6) | bit(7))
# post proc parameter length position. image_params[n..n+len-1] is for post-proc
FLAGS_POST_PROC_PARAMS_BIT_POS = 2
# length of extended parameters
FLAGS_PARAMS_HAS_EXT_LEN = (bit(16) | bit(17) | bit(18) | bit(19) | 
                            bit(20) | bit(21) | bit(22) | bit(23))
# extended parameter length position. image_params[n..n+ext_len-1] is model specific
FLAGS_EXT_LEN_BIT_POS = 16

# Configuration parameters
CONFIG_USE_FLASH_MODEL = bit(0)     # 1: use flash model, 0: use model in memory

# ISI START command config
DEFAULT_CONFIG = bit(0)
DOWNLOAD_MODEL = bit(1)

# HICO configuration structure
HICO_RESULTS_SEL_IMG = bit(0)
HICO_FLAGS_LOOP_BACK = bit(0)
HICO_FLAGS_MODEL_IN_FLASH = bit(1)

class KDPUSBSpeed(enum.IntEnum):
    """Enumeration of USB speed modes."""
    KDP_USB_SPEED_UNKNOWN = 0   # unknown USB speed
    KDP_USB_SPEED_LOW = 1       # USB low speed
    KDP_USB_SPEED_FULL = 2      # USB full speed
    KDP_USB_SPEED_HIGH = 3      # USB high speed
    KDP_USB_SPEED_SUPER = 4     # USB super speed

class KDPProductID(enum.IntEnum):
    """Enumeration of USB Product IDs."""
    KDP_DEVICE_KL520 = 0x100    # USB PID alias for KL520
    KDP_DEVICE_KL720 = 0x200    # USB PID alias for KL720

class KDPDMEConfig(KDPClass):
    """DME image configuration structure."""
    _fields_ = [("model_id", ctypes.c_int32),       # model identification ID
                ("output_num", ctypes.c_int32),     # output number
                ("image_col", ctypes.c_int32),      # column size
                                                    #   NIR8: must be multiple of 4
                                                    #   RGB565/YCbCr422: must be multiple of 2
                ("image_row", ctypes.c_int32),      # row size
                ("image_ch", ctypes.c_int32),       # channel size
                ("image_format", ctypes.c_uint32),  # image format
                ("crop_box", KDPCropBox),           # box (y1, y2, x1, x2), for future use
                ("pad_values", KDPPadValue),        # padding (t, b, l, r), for future use
                ("ext_param", ctypes.c_float * MAX_PARAMS_LEN)] # extra parameters, like threshold

    def __init__(self, model_id=0, output_num=0, image_col=0, image_row=0, image_ch=0,
                 image_format=0, crop_box=KDPCropBox(), pad_values=KDPPadValue(),
                 ext_param=[0] * MAX_PARAMS_LEN):
        self.model_id = model_id
        self.output_num = output_num
        self.image_col = image_col
        self.image_row = image_row
        self.image_ch = image_ch
        self.image_format = image_format
        self.crop_box = crop_box
        self.pad_values = pad_values
        self.ext_param = (ctypes.c_float * MAX_PARAMS_LEN)(*ext_param)

    def __repr__(self):
        extra = "\n"
        for index, val in enumerate(self.ext_param):
            if val != 0:
                extra += "".join(["param[", str(index), "]: ", str(val), "\n"])
        return ("------------------------------- DME Config --------------------------------\n"
                f"Model ID: {self.model_id}\nNumber of outputs: {self.output_num}\nImage dims "
                f"(c, w, h): ({self.image_ch}, {self.image_col}, {self.image_row})\nImage format: "
                f"{self.image_format}\n{self.crop_box}\n{self.pad_values}\nExtra params "
                f"(unspecified are 0):{extra}\n")

class KDPISIConfig(KDPClass):
    """ISI image configuration structure."""
    _fields_ = [("app_id", ctypes.c_uint32),        # application ID
                ("res_buf_size", ctypes.c_uint32),  # result buffer size
                ("image_col", ctypes.c_uint16),     # column size
                                                    #   NIR8: must be multiple of 4
                                                    #   RGB565/YCbCr422: must be multiple of 4
                ("image_row", ctypes.c_uint16),     # row_size
                ("image_format", ctypes.c_uint32),  # image format
                ("ext_param", ctypes.c_float * MAX_PARAMS_LEN)] # extra parameters, like threshold

    def __init__(self, app_id=0, res_buf_size=0, image_col=0, image_row=0, image_format=0,
                ext_param=[0]):
        self.app_id = app_id
        self.res_buf_size = res_buf_size
        self.image_col = image_col
        self.image_row = image_row
        self.image_format = image_format
        self.ext_param = (ctypes.c_float * MAX_PARAMS_LEN)(*ext_param)

    def __repr__(self):
        extra = "\n"
        for index, val in enumerate(self.ext_param):
            if val != 0:
                extra += "".join(["param[", str(index), "]: ", str(val), "\n"])
        return ("------------------------------- ISI Config --------------------------------\n"
                f"App ID: {self.app_id}\nRes buf size: {self.res_buf_size}\nImage dims (w, h): ("
                f"{self.image_col}, {self.image_row})\nImage format: {hex(self.image_format)}\n"
                f"Extra params (unspecified are 0): {extra}\n")

class KAppISIModelConfig(KDPClass):
    """ISI START configuration."""
    _fields_ = [("model_id", ctypes.c_uint32),                      # model identification ID
                ("img", KDPImgDesc),                                # image information
                ("ext_param", ctypes.c_uint32 * MAX_PARAMS_LEN)]    # extra parameters

    def __init__(self, model_id=0, img=KDPImgDesc(), ext_param=[0]):
        self.model_id = model_id
        self.img = img
        self.ext_param = (ctypes.c_uint32 * MAX_PARAMS_LEN)(*ext_param)

    def __repr__(self):
        extra = "\n"
        for index, val in enumerate(self.ext_param):
            if val != 0:
                extra += f"param[{index}]: {val}\n"
        return ("---------------------------- ISI Model Config -----------------------------\n"
                f"Model ID: {self.model_id}\n{self.img}\nExtra params (unspecified are 0): "
                f"{extra}\n")

class KAppISIData(KDPClass):
    """ISI data."""
    # struct_size will differ from C because of 'm'
    _fields_ = [("size", ctypes.c_uint32),                  # TODO
                ("version", ctypes.c_uint32),               # TODO
                ("output_size", ctypes.c_uint32),           # result buffer size
                ("config_block", ctypes.c_uint32),          # number of config blocks (max = 4)
                ("m", KAppISIModelConfig * 1)]              # model specific config data

    def __init__(self, size=0, version=0, output_size=0, config_block=0,
                 m=[KAppISIModelConfig()]):
        self.size = size
        self.version = version
        self.output_size = output_size
        self.config_block = config_block
        self.m = (KAppISIModelConfig * 1)(*m)

    def __repr__(self):
        m = "\n"
        for index, model_config in enumerate(self.m[:self.config_block]):
            m += f"Model config #{index}\n{model_config}\n"
        return ("-------------------------------- ISI Data ---------------------------------\n"
                f"Size: {self.size}\nVersion: {self.version}\nOutput size: {self.output_size}\n"
                f"Number of config blocks: {self.config_block}\n{m}\n")

class KDPISIStart(KDPClass):
    """ISI START command structure."""
    _pack_ = 4
    _fields_ = [("app_id", ctypes.c_uint32),            # application ID
                ("compatible_cfg", ctypes.c_uint32),    # TODO
                ("start_flag", ctypes.c_uint32),        # TODO
                ("config", KAppISIData)]                # TODO

    def __init__(self, app_id=0, compatible_cfg=0, start_flag=0, config=KAppISIData()):
        self.app_id = app_id
        self.compatible_cfg = compatible_cfg
        self.start_flag = start_flag
        self.config = config

    def __repr__(self):
        return ("-------------------------------- ISI Start --------------------------------\n"
                f"App ID: {self.app_id}\nCompatible config: {self.compatible_cfg}\nStart flag: "
                f"{self.start_flag}\n{self.config}\n")

class KDPISIImageHeader(KDPClass):
    """ISI image header."""
    _fields_ = [("image_length", ctypes.c_uint32),      # image length in bytes
                ("image_seq_num", ctypes.c_uint32),     # image sequence number
                ("image_col", ctypes.c_uint16),         # image pixel column
                ("image_row", ctypes.c_uint16),         # image pixel row
                ("image_format", ctypes.c_uint32),      # image format
                ("model_id", ctypes.c_uint32),          # model ID
                ("flags", ctypes.c_uint32),             # flags for using variable size of 
                                                        #   image_params
                ("image_params", ctypes.c_uint16 * 1)]  # image processing parameters in
                                                        #   units of u16 - variable size

    def __init__(self, image_length=0, image_seq_num=0, image_col=0, image_row=0, image_format=0,
                 model_id=0, flags=0, image_params=[0]):
        self.image_length = image_length
        self.image_seq_num = image_seq_num
        self.image_col = image_col
        self.image_row = image_row
        self.image_format = image_format
        self.model_id = model_id
        self.flags = flags
        self.image_params = (ctypes.c_uint16 * 1)(*image_params)

    def __repr__(self):
        # TODO - check image params and flags how they work
        return ("---------------------------- ISI Image Header -----------------------------\n"
                f"Image length: {self.image_length}\nImage sequence number: {self.image_seq_num}\n"
                f"Image dims (w, h): ({self.image_col}, {self.image_row})\nImage format: "
                f"{hex(self.image_format)}\nModel ID: {self.model_id}\nFlags: {self.flags}\nImage "
                f"processing parameters: {self.image_params[0]}\n")

class KDPNEFMetadata(KDPClass):
    """Metadata for NEF model data: metadata / fw_info / all_models."""
    _fields_ = [("platform", ctypes.c_char * 32),       # usb dongle, 96 board, etc.
                ("target", ctypes.c_uint32),            # 0: KL520, 1: KL720, etc.
                ("crc", ctypes.c_uint32),               # CRC value for all_models data
                ("kn_num", ctypes.c_uint32),            # KN number
                ("enc_type", ctypes.c_uint32),          # encrypt type
                ("tc_ver", ctypes.c_char * 32),         # toolchain version
                ("compiler_ver", ctypes.c_char * 32)]   # compiler version

    def __init__(self, platform="", target=0, crc=0, kn_num=0, enc_type=0, tc_ver="",
                 compiler_ver=""):
        self.platform = platform.encode()
        self.target = target
        self.crc = crc
        self.kn_num = kn_num
        self.enc_type = enc_type
        self.tc_ver = tc_ver.encode()
        self.compiler_ver = compiler_ver.encode()

    def __repr__(self):
        return ("------------------------------ NEF Metadata -------------------------------\n"
                f"Platform: {self.platform.decode()}\nTarget: {self.target}\nCRC: {self.crc}\n"
                f"KN number: {self.kn_num}\nEncrypt type: {self.enc_type}\nToolchain version: "
                f"{self.tc_ver.decode()}\nCompiler version: {self.compiler_ver.decode()}")

class KDPDBConfig(KDPClass):
    """DB configuration structure, FDR app only."""
    _fields_ = [("db_num", ctypes.c_uint16),    # number of database
                ("max_uid", ctypes.c_uint16),   # max number of user ID in each database
                ("max_fid", ctypes.c_uint16)]   # max number of feature maps

    def __init__(self, db_num=0, max_uid=0, max_fid=0):
        self.db_num = db_num
        self.max_uid = max_uid
        self.max_fid = max_fid

    def __repr__(self):
        return ("------------------------------- DB Config ---------------------------------\n"
                f"Number of databases: {self.db_num}\nMax uID: {self.max_uid}\n"
                f"Max fID: {self.max_fid}\n")

class KDPDeviceInfo(KDPClass):
    """Device information structure."""
    _fields_ = [("scan_index", ctypes.c_int),           # scanned order index, can be used 
                                                        #   for kdp_connect_usb_device()
                ("isConnectable", ctypes.c_bool),       # indicate if this device is connectable
                ("vendor_id", ctypes.c_ushort),         # supposed to be 0x3231
                ("product_id", ctypes.c_ushort),        # for KDPProductId
                ("link_speed", ctypes.c_int),           # for KDPUSBSpeed
                ("serial_number", ctypes.c_uint),       # KN number
                ("device_path", ctypes.c_char * 20)]    # "bus_No-hub_port_No-device_port_No"
                                                        #   ex: "1-2-3" = bus 1, (hub) port 2,
                                                        #   (device) port 3

    def __init__(self, scan_index=0, isConnectable=False, vendor_id=0, product_id=0, link_speed=0,
                 serial_number=0, device_path=""):
        self.scan_index = scan_index
        self.isConnectable = isConnectable
        self.vendor_id = vendor_id
        self.product_id = product_id
        self.link_speed = link_speed
        self.serial_number = serial_number
        self.device_path = device_path.encode()

    def __repr__(self):
        return ("------------------------------- Device Info -------------------------------\n"
                f"Scan index: {self.scan_index}\nIs connectable: {self.isConnectable}\nVendor ID: "
                f"{self.vendor_id}\nProduct ID: {self.product_id}\nLink speed: {self.link_speed}\n"
                f"Serial number: {self.serial_number}\nDevice path: {self.device_path.decode()}")

class KDPDeviceInfoList(KDPClass):
    """Information structure of connected devices."""
    # struct_size will differ from C because of 'kdevice'
    _fields_ = [("num_dev", ctypes.c_int),          # number of connected devices
                ("kdevice", KDPDeviceInfo * 10)]    # list of information on each connected device

    def __init__(self, num_dev=0, kdevice=[KDPDeviceInfo()]):
        self.num_dev = num_dev
        self.kdevice = (KDPDeviceInfo * 10)(*kdevice)

    def __repr__(self):
        devices = "\n"
        for index, device in enumerate(self.kdevice[:self.num_dev]):
            devices += f"Device #{index}:\n{device}\n"
        return ("---------------------------- Device Info List -----------------------------\n"
                f"Connected devices: {self.num_dev}\n{devices}")

# from example/KL720/post_processing_ex.h
class PostParameter720(KDPClass):
    """Parameters for postprocessing."""
    _fields_ = [("raw_input_row", ctypes.c_uint),
                ("raw_input_col", ctypes.c_uint),
                ("model_input_row", ctypes.c_uint),
                ("model_input_col", ctypes.c_uint),
                ("image_format", ctypes.c_uint),
                ("threshold", ctypes.c_float),
                ("flag", ctypes.c_uint)]

    def __init__(self, raw_input_row=0, raw_input_col=0, model_input_row=0,
                 model_input_col=0, image_format=0, threshold=0.0, flag=0):
        self.raw_input_row = raw_input_row
        self.raw_input_col = raw_input_col
        self.model_input_row = model_input_row
        self.model_input_col = model_input_col
        self.image_format = image_format
        self.threshold = threshold
        self.flag = flag

    def __repr__(self):
        return (f"Postprocess params:\nRaw input (w, h): ({self.raw_input_col}, "
                f"{self.raw_input_row})\nModel input (w, h): ({self.model_input_col}, "
                f"{self.model_input_row})\nImage format: {self.image_format}\n"
                f"Threshold: {self.threshold}\nFlag: {self.flag}")

FW_FILE_SIZE_720 = 2 * 1024 * 1024
MAX_MODEL_SIZE_720 = 80 * 1024 * 1024
FW_FILE_SIZE_520 = 128 * 1024
MAX_MODEL_SIZE_FLASH_520 = 20 * 1024 * 1024
MAX_MODEL_SIZE_DDR_520 = 40 * 1024 * 1024

SLEEP_TIME = 0.001

## ISI constants
ISI_DETECTION_SIZE = 2048
ISI_START_DATA_SIZE_720 = 1500
ISI_RAW_SIZE_520 = 768000
ISI_RESULT_SIZE_720 = 0x240000

## Default image parameters
IMAGE_SOURCE_W_DEFAULT = 640
IMAGE_SOURCE_H_DEFAULT = 480
IMAGE_SIZE_RGB565_DEFAULT = IMAGE_SOURCE_W_DEFAULT * IMAGE_SOURCE_H_DEFAULT * 2
IMAGE_SIZE_RGBA_DEFAULT = IMAGE_SOURCE_W_DEFAULT * IMAGE_SOURCE_H_DEFAULT * 4
