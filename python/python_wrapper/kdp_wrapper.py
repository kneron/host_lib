"""
This is KDP wrapper.
"""
from __future__ import absolute_import
import ctypes
import math
import sys
from time import sleep
import cv2
import numpy as np
from common import constants
import kdp_host_api as api

#from keras.applications.mobilenet_v2 import MobileNetV2
#from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
HOST_LIB_DIR = ""
TEST_DIR = "".join([HOST_LIB_DIR, "../test_images"])
TEST_DME_MOBILENET_DIR = "".join([TEST_DIR, "/dme_mobilenet/"])
DME_MODEL_FILE = "".join([TEST_DME_MOBILENET_DIR, "all_models.bin"])
DME_FW_SETUP = "".join([TEST_DME_MOBILENET_DIR, "fw_info.bin"])

TEST_DME_SSD_FD_DIR = "".join([TEST_DIR, "/dme_ssd_fd/"])
DME_SSD_FD_MODEL_FILE = "".join([TEST_DME_SSD_FD_DIR, "all_models.bin"])
DME_SSD_FD_FW_SETUP = "".join([TEST_DME_SSD_FD_DIR, "fw_info.bin"])

TEST_DME_YOLO_224_DIR = "".join([TEST_DIR, "/dme_yolo_224/"])
DME_YOLO_224_MODEL_FILE = "".join([TEST_DME_YOLO_224_DIR, "all_models.bin"])
DME_YOLO_224_FW_SETUP = "".join([TEST_DME_YOLO_224_DIR, "fw_info.bin"])

IMG_SOURCE_W = 640
IMG_SOURCE_H = 480
DME_IMG_SIZE = IMG_SOURCE_W * IMG_SOURCE_H * 2
DME_MODEL_SIZE = 20 * 1024 * 1024
DME_FWINFO_SIZE = 512
DME_SEND_IMG_RETRY_TIME = 2
SLEEP_TIME = 0.001

ISI_IMG_SIZE = IMG_SOURCE_W * IMG_SOURCE_H * 2

def pad_up_16(value):
    """Aligns value argument to 16"""
    return math.ceil(value / 16) * 16
####################################SFID####################################
FDR_IMG_SIZE      = (IMG_SOURCE_W * IMG_SOURCE_H * 2)
FDR_THRESH        = 0.475
IMG_FORMAT_RGB565 = constants.IMAGE_FORMAT_SUB128 | constants.NPU_FORMAT_RGB565
img_idx           = 0

def softmax(logits):
    """
    softmax for logits like [[[x1,x2], [y1,y2], [z1,z2], ...]]
    minimum and maximum here work as preventing overflow
    """
    clas = np.exp(np.minimum(logits, 22.))
    clas = clas / np.maximum(np.sum(clas, axis=-1, keepdims=True), 1e-10)
    return clas


def get_object_detection_res(dev_idx, inf_size, frames):
    """Gets detection results."""
    det_res = []

    inf_res = (ctypes.c_char * inf_size)()
    api.kdp_dme_retrieve_res(dev_idx, 0, inf_size, inf_res)

    od_header_res = ctypes.cast(
        ctypes.byref(inf_res), ctypes.POINTER(constants.ObjectDetectionRes)).contents
    box_count = od_header_res.box_count
    #det_res.append(od_header_res.class_count)
    #det_res.append(od_header_res.box_count)
    #print("image -> {} object(s)\n".format(box_count))

    r_size = 4
    if r_size >= 4:
        header_result = ctypes.cast(
            ctypes.byref(inf_res), ctypes.POINTER(constants.ObjectDetectionRes)).contents
        box_result = ctypes.cast(
            ctypes.byref(header_result.boxes),
            ctypes.POINTER(constants.BoundingBox * header_result.box_count)).contents
        for box in box_result:
            x1 = int(box.x1)
            y1 = int(box.y1)
            x2 = int(box.x2)
            y2 = int(box.y2)
            score = float(box.score)
            class_num = int(box.class_num)
            res = [x1, y1, x2, y2, class_num, score]
            det_res.append(res)

    return det_res
    #return np.asarray(det_res)

def get_landmark_res(dev_idx, inf_size, frames):
    """Gets landmark results."""
    inf_res = (ctypes.c_char * inf_size)()
    api.kdp_dme_retrieve_res(dev_idx, 0, inf_size, inf_res)

    lm_res = ctypes.cast(
        ctypes.byref(inf_res), ctypes.POINTER(constants.LandmakrResult)).contents
    score = lm_res.score
    blur = lm_res.blur
    print(score, blur)

    return lm_res

def get_age_gender_res(dev_idx, inf_size):
    #inf_res = (ctypes.c_char * inf_size)()
    inf_res = constants.FDAgeGenderRes()
    api.kdp_dme_retrieve_res(dev_idx, 0, inf_size, ctypes.cast(ctypes.byref(inf_res), ctypes.c_char_p))
    det_res = []
    FACE_SCORE_THRESHOLD = 0.8
    if inf_res.fd_res.score > FACE_SCORE_THRESHOLD:
        # print("[INFO] FACE DETECT (x1, y1, x2, y2, score) = {}, {}, {}, {}, {}\n".format(
        #     inf_res.fd_res.x1, inf_res.fd_res.y1, inf_res.fd_res.x2, inf_res.fd_res.y2,
        #     inf_res.fd_res.score))
        if not inf_res.ag_res.age and not inf_res.ag_res.ismale:
            #print("[INFO] FACE TOO SMALL\n")
            res = [int(inf_res.fd_res.x1), int(inf_res.fd_res.y1), int(inf_res.fd_res.x2), int(inf_res.fd_res.y2),
                float(inf_res.fd_res.score), 0, 3]  # age:0 gender:3
        else:
            #gender = "Male" if inf_res.ag_res.ismale else "Female"
            # print("[INFO] AGE_GENDER (Age, Gender) = {}, {}\n".format(
            #     inf_res.ag_res.age, gender))
            res = [int(inf_res.fd_res.x1), int(inf_res.fd_res.y1), int(inf_res.fd_res.x2), int(inf_res.fd_res.y2),
                float(inf_res.fd_res.score), int(inf_res.ag_res.age), int(inf_res.ag_res.ismale) ]  # male:1 female:2    
        det_res.append(res)        
    # else:
    #     print("[INFO] NO FACE OR FACE SCORE TOO LOW!!!\n")  

    return det_res 

def get_detection_res(dev_idx, inf_size):
    """Gets detection results."""
    inf_res = (ctypes.c_char * inf_size)()
    # Get the data for all output nodes: TOTAL_OUT_NUMBER + (H/C/W/RADIX/SCALE) +
    # (H/C/W/RADIX/SCALE) + ... + FP_DATA + FP_DATA + ...
    api.kdp_dme_retrieve_res(dev_idx, 0, inf_size, inf_res)

    # Prepare for postprocessing
    listdata = [ord(byte) for byte in inf_res]
    npdata = np.asarray(listdata)

    fp_header_res = ctypes.cast(
        ctypes.byref(inf_res), ctypes.POINTER(constants.RawFixpointData)).contents
    output_num = fp_header_res.output_num

    outnode_params_res = ctypes.cast(
        ctypes.byref(fp_header_res.out_node_params),
        ctypes.POINTER(constants.OutputNodeParams * output_num)).contents

    height = 0
    channel = 0
    width = 0
    radix = 0
    scale = 0.0
    npraw_data_array = []
    data_offset = 0
    for param in outnode_params_res:
        height = param.height
        channel = param.channel
        width = param.width
        radix = param.radix
        scale = param.scale

        # print(output_num, height, channel, width, pad_up_16(width), radix, scale)

        # offset in bytes for TOTAL_OUT_NUMBER + (H/C/W/RADIX/SCALE) + (H/C/W/RADIX/SCALE)
        offset = ctypes.sizeof(ctypes.c_int) + output_num * ctypes.sizeof(constants.OutputNodeParams)
        # print("offset ", offset, ctypes.sizeof(c_int), ctypes.sizeof(OutputNodeParams))

        # get the fixed-point data
        npdata = npdata.astype("int8")
        raw_data = []

        raw_data = npdata[offset + data_offset:offset + data_offset + height*channel*pad_up_16(width)]
        data_offset += height*channel*pad_up_16(width)
        # print(raw_data.shape, offset, offset + height*channel*pad_up_16(width), height*channel*pad_up_16(width))
        raw_data = raw_data.reshape(height, channel, pad_up_16(width))
        raw_data = raw_data[:,:,:width]

        # save the fp data into numpy array and convert to float
        npraw_data = np.array(raw_data)
        npraw_data = npraw_data.transpose(0, 2, 1) / (2 ** radix) / scale
        npraw_data_array.append(npraw_data)

    return npraw_data_array

def capture_frame(image):
    if isinstance(image, str):
        print(image)
        frame = cv2.imread(image)
        
    if isinstance(image, np.ndarray):
        frame = image

    frame = cv2.resize(frame, (IMG_SOURCE_W, IMG_SOURCE_H), interpolation=cv2.INTER_CUBIC)
    #cv2.imshow('', frame)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGR565)
    frame_data = frame.reshape(DME_IMG_SIZE)
    buf_len = DME_IMG_SIZE
    c_char_p = ctypes.POINTER(ctypes.c_char)
    frame_data = frame_data.astype(np.uint8)
    data_p = frame_data.ctypes.data_as(c_char_p)

    return data_p, buf_len

def kdp_dme_load_model(dev_idx, _model_path):
    """Load dme model."""
    model_id = 0
    data = (ctypes.c_char * DME_FWINFO_SIZE)()
    p_buf = (ctypes.c_char * DME_MODEL_SIZE)()
    ret_size = 0

    # read firmware setup data
    print("loading models to Kneron Device: ")
    n_len = api.read_file_to_buf(data, DME_FW_SETUP, DME_FWINFO_SIZE)
    if n_len <= 0:
        print("reading fw setup file failed: {}...\n".format(n_len))
        return -1

    dat_size = n_len

    n_len = api.read_file_to_buf(p_buf, DME_MODEL_FILE, DME_MODEL_SIZE)
    if n_len <= 0:
        print("reading model file failed: {}...\n".format(n_len))
        return -1

    buf_len = n_len
    model_size = n_len

    print("starting DME mode ...\n")
    ret, ret_size = api.kdp_start_dme(
        dev_idx, model_size, data, dat_size, ret_size, p_buf, buf_len)
    if ret:
        print("could not set to DME mode:{}..\n".format(ret_size))
        return -1

    print("DME mode succeeded...\n")
    print("Model loading successful")
    sleep(SLEEP_TIME)

   # dme configuration
    model_id = 1000  # model id when compiling in toolchain
    output_num = 1     # number of output node for the model
    image_col = 640
    image_row = 480
    image_ch = 3
    image_format = (constants.IMAGE_FORMAT_SUB128 |
                    constants.NPU_FORMAT_RGB565 |
                    constants.IMAGE_FORMAT_RAW_OUTPUT |
                    constants.IMAGE_FORMAT_CHANGE_ASPECT_RATIO)

    dme_cfg = constants.KDPDMEConfig(model_id, output_num, image_col,
                                     image_row, image_ch, image_format)

    dat_size = ctypes.sizeof(dme_cfg)
    print("starting DME configure ...\n")
    ret, model_id = api.kdp_dme_configure(
        dev_idx, ctypes.cast(ctypes.byref(dme_cfg), ctypes.c_char_p), dat_size, model_id)
    if ret:
        print("could not set to DME configure mode..\n")
        return -1

    print("DME configure model [{}] succeeded...\n".format(model_id))
    sleep(SLEEP_TIME)
    return 0

def kdp_dme_load_ssd_model(dev_idx, _model_path):
    """Load dme model."""
    model_id = 0
    data = (ctypes.c_char * DME_FWINFO_SIZE)()
    p_buf = (ctypes.c_char * DME_MODEL_SIZE)()
    ret_size = 0

    # read firmware setup data
    print("loading models to Kneron Device: ")
    n_len = api.read_file_to_buf(data, DME_SSD_FD_FW_SETUP, DME_FWINFO_SIZE)
    if n_len <= 0:
        print("reading fw setup file failed: {}...\n".format(n_len))
        return -1

    dat_size = n_len

    n_len = api.read_file_to_buf(p_buf, DME_SSD_FD_MODEL_FILE, DME_MODEL_SIZE)
    if n_len <= 0:
        print("reading model file failed: {}...\n".format(n_len))
        return -1

    buf_len = n_len
    model_size = n_len

    print("starting DME mode ...\n")
    ret, ret_size = api.kdp_start_dme(
        dev_idx, model_size, data, dat_size, ret_size, p_buf, buf_len)
    if ret:
        print("could not set to DME mode:{}..\n".format(ret_size))
        return -1

    print("DME mode succeeded...\n")
    print("Model loading successful")
    sleep(SLEEP_TIME)

    # dme configuration
    model_id = 3       # model id when compiling in toolchain
    output_num = 1     # number of output node for the model
    image_col = 640
    image_row = 480
    image_ch = 3
    image_format = (constants.IMAGE_FORMAT_SUB128 |
                    constants.NPU_FORMAT_RGB565)

    dme_cfg = constants.KDPDMEConfig(model_id, output_num, image_col,
                                     image_row, image_ch, image_format)

    dat_size = ctypes.sizeof(dme_cfg)
    print("starting DME configure ...\n")
    ret, model_id = api.kdp_dme_configure(
        dev_idx, ctypes.cast(ctypes.byref(dme_cfg), ctypes.c_char_p), dat_size, model_id)
    if ret:
        print("could not set to DME configure mode..\n", model_id)
        return -1

    print("DME configure model [{}] succeeded...\n".format(model_id))
    sleep(SLEEP_TIME)
    return 0

def kdp_dme_load_yolo_model(dev_idx, _model_path):
    """Load dme model."""
    model_id = 0
    data = (ctypes.c_char * DME_FWINFO_SIZE)()
    p_buf = (ctypes.c_char * DME_MODEL_SIZE)()
    ret_size = 0

    # read firmware setup data
    print("loading models to Kneron Device: ")
    n_len = api.read_file_to_buf(data, DME_YOLO_224_FW_SETUP, DME_FWINFO_SIZE)
    if n_len <= 0:
        print("reading fw setup file failed: {}...\n".format(n_len))
        return -1

    dat_size = n_len

    n_len = api.read_file_to_buf(p_buf, DME_YOLO_224_MODEL_FILE, DME_MODEL_SIZE)
    if n_len <= 0:
        print("reading model file failed: {}...\n".format(n_len))
        return -1

    buf_len = n_len
    model_size = n_len

    print("starting DME mode ...\n")
    ret, ret_size = api.kdp_start_dme(
        dev_idx, model_size, data, dat_size, ret_size, p_buf, buf_len)
    if ret:
        print("could not set to DME mode:{}..\n".format(ret_size))
        return -1

    print("DME mode succeeded...\n")
    print("Model loading successful")
    sleep(SLEEP_TIME)

    # dme configuration
    model_id = 19      # model id when compiling in toolchain
    output_num = 2     # number of output node for the model
    image_col = 640
    image_row = 480
    image_ch = 3
    image_format = (constants.IMAGE_FORMAT_SUB128 |
                    constants.NPU_FORMAT_RGB565 |
                    constants.IMAGE_FORMAT_RAW_OUTPUT)

    dme_cfg = constants.KDPDMEConfig(model_id, output_num, image_col,
                                     image_row, image_ch, image_format)

    dat_size = ctypes.sizeof(dme_cfg)
    print("starting DME configure ...\n")
    ret, model_id = api.kdp_dme_configure(
        dev_idx, ctypes.cast(ctypes.byref(dme_cfg), ctypes.c_char_p), dat_size, model_id)
    if ret:
        print("could not set to DME configure mode..\n", model_id)
        return -1

    print("DME configure model [{}] succeeded...\n".format(model_id))
    sleep(SLEEP_TIME)
    return 0

def kdp_dme_load_age_gender_model(dev_idx, _model_path):
    """Load dme model."""
    model_id = 0
    data = (ctypes.c_char * DME_FWINFO_SIZE)()
    p_buf = (ctypes.c_char * DME_MODEL_SIZE)()
    ret_size = 0

    model_file = _model_path+"all_models.bin"
    fw_setup = _model_path+"fw_info.bin"

    # read firmware setup data
    print("loading models to Kneron Device: ")
    n_len = api.read_file_to_buf(data, fw_setup, DME_FWINFO_SIZE)
    if n_len <= 0:
        print("reading fw setup file failed: {}...\n".format(n_len))
        return -1

    dat_size = n_len

    n_len = api.read_file_to_buf(p_buf, model_file, DME_MODEL_SIZE)
    if n_len <= 0:
        print("reading model file failed: {}...\n".format(n_len))
        return -1

    buf_len = n_len
    model_size = n_len

    print("starting DME mode ...\n")
    ret, ret_size = api.kdp_start_dme(
        dev_idx, model_size, data, dat_size, ret_size, p_buf, buf_len)
    if ret:
        print("could not set to DME mode:{}..\n".format(ret_size))
        return -1

    print("DME mode succeeded...\n")
    print("Model loading successful")
    sleep(SLEEP_TIME)

    # dme configuration
    model_id = 3       # model id when compiling in toolchain
    output_num = 1     # number of output node for the model
    image_col = 640
    image_row = 480
    image_ch = 3
    image_format = (constants.IMAGE_FORMAT_MODEL_AGE_GENDER |
                    constants.IMAGE_FORMAT_SUB128 |
                    constants.NPU_FORMAT_RGB565)

    dme_cfg = constants.KDPDMEConfig(model_id, output_num, image_col,
                                     image_row, image_ch, image_format)

    dat_size = ctypes.sizeof(dme_cfg)
    print("starting DME configure ...\n")
    ret, model_id = api.kdp_dme_configure(
        dev_idx, ctypes.cast(ctypes.byref(dme_cfg), ctypes.c_char_p), dat_size, model_id)
    if ret:
        print("could not set to DME configure mode..\n", model_id)
        return -1

    print("DME configure model [{}] succeeded...\n".format(model_id))
    sleep(SLEEP_TIME)
    return 0    

def sync_inference(device_index, app_id, input_size, capture,
                  img_id_tx, frames, post_handler):
    """Send the rest of images and get the results.

    Arguments:
        device_index: Connected device ID. A host can connect several devices.
        app_id: ID of application to be run.
        input_size: Size of input image.
        ret_size: Return size.
        capture: Active cv2 video capture instance.
        img_id_tx: Should be returned from fill_buffer.
        frames: List of frames captured by the video capture instance.
        post_handler: Function to process the results of the inference.
    """
    ret_size = 2048
    inf_res = (ctypes.c_char * ret_size)()

    data_p = isi_capture_frame(capture, frames)

    ret, _, img_left = isi_inference(
        device_index, data_p, input_size, img_id_tx, 0, 0)
    if ret:
        return ret

    _, _, result_size = isi_get_result(
        device_index, img_id_tx, 0, 0, inf_res, app_id)

    post_handler(inf_res, result_size, frames)

    return


def kdp_inference(dev_idx, img_path):
    """Performs dme inference."""
    img_buf, buf_len = capture_frame(img_path)
    inf_size = 0
    inf_res = (ctypes.c_char * 256000)()
    res_flag = False
    mode = 1
    model_id = 0
    ssid = 0
    status = 0
    _ret, ssid, res_flag = api.kdp_dme_inference(
        dev_idx, img_buf, buf_len, ssid, res_flag, inf_res, mode, model_id)
    # get status for session 1
    while 1:
        status = 0  # Must re-initialize status to 0
        _ret, ssid, status, inf_size = api.kdp_dme_get_status(
            dev_idx, ssid, status, inf_size, inf_res)
        # print(status, inf_size)
        if status == 1:
            npraw_data = get_detection_res(dev_idx, inf_size)
            break
    return npraw_data

def kdp_dme_inference(dev_idx, app_id, capture, buf_len, frames):
    """Performs dme inference."""
    img_buf = isi_capture_frame(capture, frames)
    inf_size = 0
    inf_res = (ctypes.c_char * 256000)()
    res_flag = False
    mode = 0
    model_id = 0

    _ret, inf_size, res_flag = api.kdp_dme_inference(
        dev_idx, img_buf, buf_len, inf_size, res_flag, inf_res, mode, model_id)

    if (app_id == constants.APP_AGE_GENDER):
        det_res = get_age_gender_res(dev_idx, inf_size)
    elif (app_id == constants.APP_FD_LM):
        det_res = get_object_detection_res(dev_idx, inf_size, frames)
    elif (app_id == constants.APP_TINY_YOLO3):
        det_res = get_detection_res(dev_idx, inf_size)

    return det_res  

def kdp_exit_dme(dev_idx):
    api.kdp_end_dme(dev_idx)

def load_reg_user_list(reg_user_list):
    list_path = './data/fdr/userlist.txt'
    np_list = np.loadtxt(list_path).astype(int)
    if (np_list.size == 20):
        reg_user_list = np_list.tolist()
    return reg_user_list

def save_reg_user_list(reg_user_list):
    list_path = './data/fdr/userlist.txt'
    np_list = np.array(reg_user_list).astype(int)
    np.savetxt(list_path,np_list,fmt='%i')

def capture_cam_frame(cap):
    cv_ret, frame1 = cap.read()
    frame1 = cv2.flip(frame1, 1)

    return frame1

def frame_to_565_data(frame1):
    frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2BGR565)
    frame_data = frame.reshape(FDR_IMG_SIZE)
    c_char_p = ctypes.POINTER(ctypes.c_char)
    frame_data = frame_data.astype(np.uint8)
    data_p = frame_data.ctypes.data_as(c_char_p)

    return data_p

def start_reg_mode(dev_idx, user_id):
    img_size = 0
    ret, img_size = api.kdp_start_sfid_mode(dev_idx, img_size, FDR_THRESH, IMG_SOURCE_W, IMG_SOURCE_H, IMG_FORMAT_RGB565)

    if (ret != 0) or (img_size == 0):
        print("start verify mode")
        return -1

    sleep(SLEEP_TIME)
    global img_idx
    img_idx += 1

    ret = api.kdp_start_reg_user_mode(dev_idx, user_id, img_idx)
    return ret

def register_user(dev_idx, frame, user_id):
    data_p = frame_to_565_data(frame)

    res = (ctypes.c_char * 0)()
    ret, mask = api.kdp_extract_feature_generic(dev_idx, data_p, FDR_IMG_SIZE, 0, res)

    if (ret):
        if (ret == constants.MSG_APP_UID_EXIST):
            print("> user exist <")
        return ret

    ret = api.kdp_register_user(dev_idx, user_id)
    if (ret):
        print("register user failed")
    return ret

def del_user_id(dev_idx, user_id):
    img_size = 0
    ret, img_size = api.kdp_start_sfid_mode(dev_idx, img_size, FDR_THRESH, IMG_SOURCE_W, IMG_SOURCE_H, IMG_FORMAT_RGB565)

    if (ret != 0) or (img_size == 0):
        print("start verify mode")
        return -1

    sleep(SLEEP_TIME)

    ret = api.kdp_remove_user(dev_idx, user_id)
    return ret

def start_inf_mode(dev_idx):
    img_size = 0
    ret, img_size = api.kdp_start_sfid_mode(dev_idx, img_size, FDR_THRESH, IMG_SOURCE_W, IMG_SOURCE_H, IMG_FORMAT_RGB565)

    if (ret != 0) or (img_size == 0):
        print("start inf mode fail")
        return -1

    return ret

def verify_user_id(dev_idx, frame):
    mask = api.kdp_get_res_mask(1, 1, 0, 0)
    res_size = api.kdp_get_res_size(1, 1, 0, 0)
    res = (ctypes.c_char * res_size)()
    user_id = mask

    data_p = frame_to_565_data(frame)

    ret, u_id, mask_value = api.kdp_verify_user_id_generic(dev_idx, user_id, data_p, FDR_IMG_SIZE, mask, res)

    fd_lm_res = ctypes.cast(ctypes.byref(res), ctypes.POINTER(constants.FDLMRes)).contents
    x = fd_lm_res.fd_res.x
    y = fd_lm_res.fd_res.y
    w = fd_lm_res.fd_res.w
    h = fd_lm_res.fd_res.h

    return u_id, x, y, w, h

def isi_inference(dev_idx, img_buf, buf_len, img_id, rsp_code, window_left):
    """Performs ISI inference.

    Arguments:
        device_index: Connected device ID. A host can connect several devices.
        img_buf: Image buffer.
        buf_len: File size.
        img_id: Sequence ID of the image.
        rsp_code:
        window_left: Number of image buffers still available for input.
    """
    ret, rsp_code, window_left = api.kdp_isi_inference(
        dev_idx, img_buf, buf_len, img_id, rsp_code, window_left)
    if ret:
        print("ISI inference failed: {}\n".format(ret))
        return -1
    if rsp_code:
        print("ISI inference error_code: [{}] [{}]\n".format(rsp_code, window_left))
        return -1

    return ret, rsp_code, window_left

def isi_get_result(dev_idx, img_id, rsp_code, r_size, r_data, app_id):
    """Gets inference results.

    Arguments:
        dev_idx: Connected device ID. A host can connect several devices.
        img_id: Sequence ID to get inference results of an image with that ID.
        rsp_code:
        r_size: Inference data size.
        r_data: Inference result data.
        app_id: ID of application to be run.
    """
    ret, rsp_code, r_size = api.kdp_isi_retrieve_res(dev_idx, img_id, rsp_code, r_size, r_data)
    if ret:
        print("ISI get [{}] result failed: {}\n".format(img_id, ret))
        return -1, rsp_code, r_size

    if rsp_code:
        print("ISI get [{}] result error_code: [{}] [{}]\n".format(img_id, rsp_code, r_size))
        return -1, rsp_code, r_size

    if r_size >= 4:
        if app_id == constants.APP_AGE_GENDER: # age_gender
            gender = ["Female", "Male"]
            result = ctypes.cast(
                ctypes.byref(r_data), ctypes.POINTER(constants.FDAgeGenderS)).contents
            box_count = result.count
            print("Img [{}]: {} people\n".format(img_id, box_count))
            box = ctypes.cast(
                ctypes.byref(result.boxes),
                ctypes.POINTER(constants.FDAgeGenderRes * box_count)).contents

            for idx in range(box_count):
                print("[{}]: {}, {}\n".format(idx, gender[box[idx].ag_res.ismale], box[idx].ag_res.age))
        else: # od, yolo
            od_header_res = ctypes.cast(
                ctypes.byref(r_data), ctypes.POINTER(constants.ObjectDetectionRes)).contents
            box_count = od_header_res.box_count
            print("image {} -> {} object(s)\n".format(img_id, box_count))

        return 0, rsp_code, r_size
    print("Img [{}]: result_size {} too small\n".format(img_id, r_size))
    return -1, rsp_code, r_size

def isi_capture_frame(cap, frames):
    """Frame read and convert to RGB565.

    Arguments:
        cap: Active cv2 video capture instance.
        frames: List of frames for the video capture to add to.
    """
    _cv_ret, frame = cap.read()
    if frame is None:
        print("fail to read from cam!")
    frame = cv2.flip(frame, 1)
    frames.append(frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGR565)
    frame_data = frame.reshape(ISI_IMG_SIZE)
    c_char_p = ctypes.POINTER(ctypes.c_char)
    frame_data = frame_data.astype(np.uint8)
    data_p = frame_data.ctypes.data_as(c_char_p)

    return data_p

def setup_capture(cam_id, width, height):
    """Sets up the video capture device.

    Returns the video capture instance on success and None on failure.

    Arguments:
        width: Width of frames to capture.
        height: Height of frames to capture.
    """
    capture = cv2.VideoCapture(cam_id)
    if not capture.isOpened():
        print("Could not open video device!")
        return None
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return capture

def start_isi(device_index, app_id, width, height):
    """Starts the ISI mode.

    Returns 0 on success and -1 on failure.

    Arguments:
        device_index: Connected device ID. A host can connect several devices.
        app_id: ID of application to be run.
        size: Return size.
        width: Width of the input image.
        height: Height of the input image.
        image_format: Format of input image.
    """
    print("starting ISI mode...\n")
    if (app_id == constants.APP_OD):
        image_format = 0x80000060 | constants.IMAGE_FORMAT_CHANGE_ASPECT_RATIO  #RGB565, no parallel mode
    else:
        image_format = 0x80000060                                               #RGB565, no parallel mode
    size = 2048

    ret, _, image_buf_size = api.kdp_start_isi_mode(
        device_index, app_id, size, width, height, image_format, 0, 0)
    if ret:
        print("could not set to ISI mode: {} ..\n".format(ret))
        return -1
    if image_buf_size < 3:
        print("ISI mode window {} too small...\n".format(image_buf_size))
        return -1

    print("ISI mode succeeded (window = {})...\n".format(image_buf_size))
    sleep(SLEEP_TIME)
    return 0

def start_isi_parallel(device_index, app_id, width, height):
    """Starts the ISI mode.

    Returns 0 on success and -1 on failure.

    Arguments:
        device_index: Connected device ID. A host can connect several devices.
        app_id: ID of application to be run.
        size: Return size.
        width: Width of the input image.
        height: Height of the input image.
        image_format: Format of input image.
    """
    print("starting ISI mode...\n")
    if (app_id == constants.APP_OD):
        image_format = 0x88000060 | constants.IMAGE_FORMAT_CHANGE_ASPECT_RATIO  #RGB565, parallel mode
    else:
        image_format = 0x88000060                                               #RGB565, parallel mode
    size = 2048

    ret, _, image_buf_size = api.kdp_start_isi_mode(
        device_index, app_id, size, width, height, image_format, 0, 0)
    if ret:
        print("could not set to ISI mode: {} ..\n".format(ret))
        return -1
    if image_buf_size < 3:
        print("ISI mode window {} too small...\n".format(image_buf_size))
        return -1

    print("ISI mode succeeded (window = {})...\n".format(image_buf_size))
    sleep(SLEEP_TIME)
    return 0    

def fill_buffer(device_index, capture, size, frames):
    """Fill up the image buffer using the capture device.

    Arguments:
        device_index: Connected device ID. A host can connect several devices.
        capture: Active cv2 video capture instance.
        size: Size of the input images.
        frames: List of frames captured by the video capture instance.
    """
    print("starting ISI inference ...\n")
    img_id_tx = 1234
    img_left = 12
    buffer_depth = 0
    while 1:
        data_p = isi_capture_frame(capture, frames)
        ret, error_code, img_left = isi_inference(
            device_index, data_p, size, img_id_tx, 0, img_left)
        if ret:
            print("Companion inference failed")
            return -1, img_id_tx, img_left, buffer_depth
        if not error_code:
            img_id_tx += 1
            buffer_depth += 1
            if not img_left:
                break
    return 0, img_id_tx, img_left, buffer_depth

def pipeline_inference(device_index, app_id, loops, input_size, capture,
                  img_id_tx, img_left, buffer_depth, frames, post_handler):
    """Send the rest of images and get the results.

    Arguments:
        device_index: Connected device ID. A host can connect several devices.
        app_id: ID of application to be run.
        loops: Number of images to get results.
        input_size: Size of input image.
        ret_size: Return size.
        capture: Active cv2 video capture instance.
        img_id_tx: Should be returned from fill_buffer.
        img_left: Should be returned from fill_buffer.
        buffer_depth: Should be returned from fill_buffer.
        frames: List of frames captured by the video capture instance.
        post_handler: Function to process the results of the inference.
    """
    img_id_rx = 1234
    ret_size = 2048
    inf_res = (ctypes.c_char * ret_size)()
    while loops:
        _, _, result_size = isi_get_result(
            device_index, img_id_rx, 0, 0, inf_res, app_id)
        post_handler(inf_res, result_size, frames)

        img_id_rx += 1
        data_p = isi_capture_frame(capture, frames)

        ret, _, img_left = isi_inference(
            device_index, data_p, input_size, img_id_tx, 0, img_left)
        if ret:
            return ret
        img_id_tx += 1
        loops -= 1

    # Get last 2 results
    while buffer_depth:
        ret, _, result_size = isi_get_result(
            device_index, img_id_rx, 0, 0, inf_res, app_id)
        post_handler(inf_res, result_size, frames)
        img_id_rx += 1
        buffer_depth -= 1
    return 0

def dme_fill_buffer(device_index, capture, size, frames):
    """Send 1 image to the DME image buffers using the capture device.

    Arguments:
        device_index: Connected device ID. A host can connect several devices.
        capture: Active cv2 video capture instance.
        size: Size of the input images.
        frames: List of frames captured by the video capture instance.
    """
    print("starting DME inference ...\n")
    inf_res = (ctypes.c_char * 256000)()
    res_flag = False
    mode = 1
    model_id = 0
    ssid = 0

    img_buf = isi_capture_frame(capture, frames)
    _ret, ssid, res_flag = api.kdp_dme_inference(
        device_index, img_buf, size, ssid, res_flag, inf_res, mode, model_id)

    return 0, ssid

def dme_pipeline_inference(device_index, app_id, loops, input_size, capture,
                  prev_ssid, frames, post_handler):
    """Send the rest of images and get the results.

    Arguments:
        device_index: Connected device ID. A host can connect several devices.
        app_id: ID of application to be run.
        loops: Number of images to get results.
        input_size: Size of input image.
        capture: Active cv2 video capture instance.
        prev_ssid: Should be returned from dme_fill_buffer.
        frames: List of frames captured by the video capture instance.
        post_handler: Function to process the results of the inference.
    """
    inf_res = (ctypes.c_char * 256000)()
    res_flag = False
    mode = 1
    model_id = 0
    ssid = 0
    inf_size = 0

    while loops:
        img_buf = isi_capture_frame(capture, frames)
        _ret, ssid, res_flag = api.kdp_dme_inference(
            device_index, img_buf, input_size, ssid, res_flag, inf_res, mode, model_id)

        # get status for previous session
        # print("ssid prev ", ssid, prev_ssid)
        while 1:
            status = 0  # Must re-initialize status to 0
            _ret, prev_ssid, status, inf_size = api.kdp_dme_get_status(
                device_index, prev_ssid, status, inf_size, inf_res)
            # print(status, inf_size)
            if status == 1:
                if (app_id == constants.APP_TINY_YOLO3):
                    npraw_data = get_detection_res(device_index, inf_size)
                    post_handler(device_index, npraw_data, frames)
                break

        prev_ssid = ssid
        loops -= 1

    # Get last 1 results
    while 1:
        status = 0  # Must re-initialize status to 0
        _ret, prev_ssid, status, inf_size = api.kdp_dme_get_status(
            device_index, prev_ssid, status, inf_size, inf_res)
        # print(status, inf_size)
        if status == 1:
            if (app_id == constants.APP_TINY_YOLO3):
                npraw_data = get_detection_res(device_index, inf_size)
                post_handler(device_index, npraw_data, frames)
            break

    return 0

def read_file_to_buf(image_file, image_size):
    """Reads input image into a buffer.

    Arguments:
        image_file: File containing the input image.
        image_size: Size of the input image.
    """
    buffer = (ctypes.c_char * image_size)()
    length = api.read_file_to_buf(buffer, image_file, image_size)
    if length <= 0:
        print("reading image file, {}, failed: {}...\n".format(image_file, length))
        return None
    return buffer

def isi_send_first_two(dev_idx, buffer, buffer_t, size):
    """Sends two images first for inference.

    Arguments:
        dev_idx: Connected device ID. A host can connect several devices.
        buffer: Buffer holding the image data from the first file.
        buffer_t: Buffer holding the image data from the second file.
        size: Size of one input image.
    """
    print("starting ISI inference ...\n")
    img_id_tx = 1234
    img_left = 12

    ret, _, img_left = isi_inference(
        dev_idx, buffer, size, img_id_tx, 0, img_left)
    if ret:
        return ret, img_left
    img_id_tx += 1

    ret, _, img_left = isi_inference(
        dev_idx, buffer_t, size, img_id_tx, 0, img_left)
    if ret:
        return ret, img_left
    img_id_tx += 1
    return 0, img_left

def isi_send_rest(dev_idx, app_id, buffer, buffer_t, input_size,
                  ret_size, img_left, test_loop):
    """Sends rest of the images for inference and results.

    Arguments:
        dev_idx: Connected device ID. A host can connect several devices.
        app_id: ID of application to be run.
        buffer: Buffer holding the image data from the first file.
        buffer_t: Buffer holding the image data from the second file.
        input_size: Size of one input image.
        ret_size: Return size.
        img_left: Number of image buffers still available for input.
        test_loop: Number of loops to send two images.
    """
    img_id_tx = 1236
    img_id_rx = 1234
    inf_res = (ctypes.c_char * ret_size)()
    loop = 0
    if test_loop > 3:
        loop = test_loop - 2

    while loop:
        ret, _, img_left = isi_inference(
            dev_idx, buffer, input_size, img_id_tx, 0, img_left)
        if ret:
            return ret, img_id_rx
        img_id_tx += 1

        ret, _, _ = isi_get_result(
            dev_idx, img_id_rx, 0, 0, inf_res, app_id)
        if ret:
            return ret, img_id_rx
        img_id_rx += 1

        loop -= 1
        # Odd loop case
        if not loop:
            break

        ret, _, img_left = isi_inference(
            dev_idx, buffer_t, input_size, img_id_tx, 0, img_left)
        if ret:
            return ret, img_id_rx
        img_id_tx += 1

        ret, _, _ = isi_get_result(
            dev_idx, img_id_rx, 0, 0, inf_res, app_id)
        if ret:
            return ret, img_id_rx
        img_id_rx += 1
        loop -= 1

    return 0, img_id_rx

def isi_get_last_results(dev_idx, app_id, img_id_rx, ret_size):
    """Gets results for last two images.

    Arguments:
        dev_idx: Connected device ID. A host can connect several devices.
        app_id: ID of application to be run.
        img_id_rx: Sequence ID to get inference results of an image with that ID
        ret_size: Return size.
    """
    inf_res = (ctypes.c_char * ret_size)()
    ret, _, _ = isi_get_result(
        dev_idx, img_id_rx, 0, 0, inf_res, app_id)
    if ret:
        return ret
    img_id_rx += 1

    ret, _, _ = isi_get_result(
        dev_idx, img_id_rx, 0, 0, inf_res, app_id)
    if ret:
        return ret
    img_id_rx += 1

    return 0
