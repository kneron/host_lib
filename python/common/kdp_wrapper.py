"""
This is KDP wrapper.
"""
from __future__ import absolute_import
import ctypes
import math
import struct
import sys
from time import sleep

import cv2
import numpy as np

from common import constants
import kdp_host_api as api

g_results = []
img_idx = 0

def pad_up_16(value):
    """Aligns value argument to 16."""
    return math.ceil(value / 16) * 16

def sigmoid(x):
    """
    sigmoid for numpy array
    """
    return 1 / (1 + np.exp(-x))

def softmax(logits):
    """
    softmax for logits like [[[x1,x2], [y1,y2], [z1,z2], ...]]
    minimum and maximum here work as preventing overflow
    """
    clas = np.exp(np.minimum(logits, 22.))
    clas = clas / np.maximum(np.sum(clas, axis=-1, keepdims=True), 1e-10)
    return clas

def dme_get_result(dev_idx, inf_size, app_id):
    """Gets inference results.

    Arguments:
        dev_idx: Integer connected device ID
        inf_size: Integer inference data size
        app_id: Integer ID of application to be run
    """
    inf_res = (ctypes.c_char * inf_size)()
    ret = api.kdp_dme_retrieve_res(dev_idx, 0, inf_size, inf_res)
    if ret:
        print("DME get result failed: {}\n".format(ret))
        return -1

    if (app_id == 0): # raw output
        # Prepare for postprocessing
        listdata = [ord(byte) for byte in inf_res]
        npdata = np.asarray(listdata)

        fp_header_res = ctypes.cast(
            ctypes.byref(inf_res), ctypes.POINTER(constants.RawFixpointData)).contents
        output_num = fp_header_res.output_num

        outnode_params_res = ctypes.cast(
            ctypes.byref(fp_header_res.out_node_params),
            ctypes.POINTER(constants.OutputNodeParams * output_num)).contents

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

            raw_data = npdata[offset + data_offset:offset + data_offset + height * channel * pad_up_16(width)]
            data_offset += height * channel * pad_up_16(width)
            # print(raw_data.shape, offset, offset + height*channel*pad_up_16(width), height*channel*pad_up_16(width))
            raw_data = raw_data.reshape(height, channel, pad_up_16(width))
            raw_data = raw_data[:, :, :width]

            # save the fp data into numpy array and convert to float
            npraw_data = np.array(raw_data)
            npraw_data = npraw_data.transpose(0, 2, 1) / (2 ** radix) / scale
            npraw_data_array.append(npraw_data)

        return npraw_data_array
    elif app_id == constants.AppID.APP_AGE_GENDER: # age_gender
        result = cast_and_get(inf_res, constants.FDAgeGenderRes)
        det_res = []
        FACE_SCORE_THRESHOLD = 0.8
        if result.fd_res.score > FACE_SCORE_THRESHOLD:
            # print("[INFO] FACE DETECT (x1, y1, x2, y2, score) = {}, {}, {}, {}, {}\n".format(
            #     result.fd_res.x1, result.fd_res.y1, result.fd_res.x2, result.fd_res.y2,
            #     result.fd_res.score))
            if not result.ag_res.age and not result.ag_res.ismale:
                # print("[INFO] FACE TOO SMALL\n")
                res = [int(result.fd_res.x1), int(result.fd_res.y1), int(result.fd_res.x2), int(result.fd_res.y2),
                       float(result.fd_res.score), 0, 3]  # age:0 gender:3
            else:
                # gender = "Male" if result.ag_res.ismale else "Female"
                # print("[INFO] AGE_GENDER (Age, Gender) = {}, {}\n".format(
                #     result.ag_res.age, gender))
                res = [int(result.fd_res.x1), int(result.fd_res.y1), int(result.fd_res.x2), int(result.fd_res.y2),
                       float(result.fd_res.score), int(result.ag_res.age),
                       int(result.ag_res.ismale)]
            det_res.append(res)
        return det_res
    else: # od, yolo
        od_header_res = cast_and_get(inf_res, constants.ObjectDetectionRes)
        det_res = []

        r_size = 4
        if r_size >= 4:
            box_result = cast_and_get(od_header_res.boxes, constants.BoundingBox * od_header_res.box_count)

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

def convert_numpy_to_char_p(frame, color=None, size=constants.IMAGE_SIZE_RGB565_DEFAULT):
    """Converts NumPy array into ctypes char pointer.

    Arguments:
        frame: NumPy array from image
        color: Integer indicating color conversion
        size: Integer size of the frame array
    """
    new_frame = frame
    if color is not None:
        new_frame = cv2.cvtColor(frame, color)
    new_frame = new_frame.reshape(size)
    new_frame = new_frame.astype(np.uint8)
    return new_frame.ctypes.data_as(ctypes.POINTER(ctypes.c_char))

def capture_frame(image):
    """Converts input image into default sizes and into ctypes data.

    Arguments:
        image: String path to input image or NumPy image array
    """
    if isinstance(image, str):
        print(image)
        frame = cv2.imread(image)
    elif isinstance(image, np.ndarray):
        frame = image

    frame = cv2.resize(frame, (constants.IMAGE_SOURCE_W_DEFAULT, constants.IMAGE_SOURCE_H_DEFAULT),
                       interpolation=cv2.INTER_CUBIC)
    return convert_numpy_to_char_p(frame, cv2.COLOR_BGR2BGR565)

def init_dme_config(model_id, output_num, image_format, image_col=constants.IMAGE_SOURCE_W_DEFAULT,
                    image_row=constants.IMAGE_SOURCE_H_DEFAULT, image_ch=3, ext_param=[0.0]):
    """Initialize DME config instance for configuration.

    Returns initialized KDPDMEConfig instance.

    Arguments:
        model_id: Integer model ID to be inferenced
        output_num: Integer number of model outputs
        image_format: Integer format of input image
        image_col: Integer width of input image
        image_row: Integer height of input image
        image_ch: Integer number of channels in input image
        ext_param: List of postprocess parameters
    """
    return constants.KDPDMEConfig(
        model_id=model_id, output_num=output_num, image_col=image_col, image_row=image_row,
        image_ch=image_ch, image_format=image_format, ext_param=ext_param)

def dme_configure(device_index, dme_config, to_print=True):
    """Change the DME configurations.

    Arguments:
        device_index: Integer connected device ID
        dme_config: KDPDMEConfig instance
        to_print: Flag to enable prints
    """
    if to_print:
        print("\nStarting DME configure...")

    dme_config_p = ctypes.cast(ctypes.pointer(dme_config), ctypes.POINTER(ctypes.c_char))
    ret_model = ctypes.c_uint32(0)
    ret = api.kdp_dme_configure(
        device_index, dme_config_p, dme_config.struct_size(), ctypes.byref(ret_model))

    if ret:
        print(f"Could not set to DME configure mode: {ret_model.value}...")
        return ret

    if to_print:
        print(f"DME configure model [{ret_model.value}] succeeded...\n")
    sleep(constants.SLEEP_TIME)

    return 0

def dme_load_model(device_index, model_file, dme_config):
    """Loads the model and sets DME configurations.

    Returns 0 on success and error code on failure.

    Arguments:
        device_index: Integer connected device ID
        model_file: String path to model
        dme_config: KDPDMEConfig instance
    """
    print("Loading models to Kneron Device: ")

    p_buf, n_len = read_file_to_buf_with_size(model_file, constants.MAX_MODEL_SIZE_720)
    if p_buf is None:
        return -1

    print("\nStarting DME mode...")

    ret_size = ctypes.c_uint32(0)
    ret = api.kdp_start_dme_ext(device_index, p_buf, n_len, ctypes.byref(ret_size))
    if ret:
        print(f"Could not set to DME mode: {ret_size}...")
        return ret

    print("\nDME mode succeeded...")
    print("Model loading successful")
    sleep(constants.SLEEP_TIME)

    ret = dme_configure(device_index, dme_config)
    if ret:
        return ret

    return 0

def sync_inference(device_index, app_id, input_size, capture,
                   img_id_tx, frames, post_handler):
    """Performs image inference on the captured frame.

    Returns 0 on success and -1 on failure.

    Arguments:
        device_index: Integer connected device ID
        app_id: Integer ID of application to be run
        input_size: Integer size of input image
        capture: Active cv2 video capture instance
        img_id_tx: Integer current image ID, should be returned from isi_fill_buffer()
        frames: List of frames captured by the video capture instance
        post_handler: Function to process the results of the inference
    """
    inf_res = (ctypes.c_char * constants.ISI_DETECTION_SIZE)()

    data_p = isi_capture_frame(capture, frames)

    img_left = isi_inference(device_index, data_p, input_size, img_id_tx)
    if img_left == -1:
        return -1

    _, result_size = isi_get_result(device_index, img_id_tx, inf_res, app_id)

    post_handler(device_index, inf_res, result_size, frames)

    return 0

def dme_inference(device_index, mode, input_image, model_id=0, app_id=0, buf_len=0, frames=[]):
    """Performs DME inference.

    Returns DME results on success and None on failure.

    Arguments:
        device_index: Integer connected device ID
        mode: Integer running mode, 0 for 'serial' and 1 for 'async'
        input_image: Active cv2 video capture instance or string path to input image
        model_id: Integer model ID to perform inference
        app_id: Integer application ID to get result from, only used if mode = 0
        buf_len: Integer length of image size, only used if input_image is cv2 capture
        frames: List of frames for the video capture to add to, only used if input_image
            is cv2 capture
    """
    inf_size = ctypes.c_uint32(0)
    res_flag = ctypes.c_bool(False)
    inf_res = (ctypes.c_char * 256000)()

    if isinstance(input_image, cv2.VideoCapture):
        img_buf = isi_capture_frame(input_image, frames)
    else:
        buf_len = constants.IMAGE_SIZE_RGB565_DEFAULT
        img_buf = capture_frame(input_image)

    ret = api.kdp_dme_inference(device_index, img_buf, buf_len, ctypes.byref(inf_size),
                                ctypes.byref(res_flag), inf_res, mode, model_id)
    if ret:
        print(f"DME inference failed: {ret}...")
        return None

    if mode == 1:   # async
        # get status for session
        while 1:
            status = ctypes.c_uint16(0)     # must re-initialize status to 0
            ssid = ctypes.c_uint16(inf_size.value)   # inf_size for async = session ID
            ret = api.kdp_dme_get_status(device_index, ctypes.byref(ssid), ctypes.byref(status),
                                         ctypes.byref(inf_size), inf_res)

            if ret:
                print(f"DME get status failed: {ret}...")
                return None

            if status.value == 1:
                return dme_get_result(device_index, inf_size.value, 0)

    # serial
    return dme_get_result(device_index, inf_size.value, app_id)

def end_det(dev_idx):
    """Ends DME mode for the specified device."""
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

    if sys.platform == "darwin":
        frame1 = cv2.resize(frame1, (constants.IMAGE_SOURCE_W_DEFAULT, constants.IMAGE_SOURCE_H_DEFAULT), interpolation=cv2.INTER_CUBIC)

    frame1 = cv2.flip(frame1, 1)

    return frame1

def start_reg_mode(dev_idx, user_id):
    img_size = 0
    img_format_rgb565 = constants.IMAGE_FORMAT_SUB128 | constants.NPU_FORMAT_RGB565
    ret, img_size = api.kdp_start_sfid_mode(dev_idx, img_size, constants.FID_THRESHOLD, constants.IMAGE_SOURCE_W_DEFAULT, constants.IMAGE_SOURCE_H_DEFAULT, img_format_rgb565)

    if (ret != 0) or (img_size == 0):
        print("start verify mode")
        return -1

    sleep(constants.SLEEP_TIME)
    global img_idx
    img_idx += 1

    ret = api.kdp_start_reg_user_mode(dev_idx, user_id, img_idx)
    return ret

def register_user(dev_idx, frame, user_id):
    data_p = convert_numpy_to_char_p(frame, cv2.COLOR_BGR2BGR565)

    res = (ctypes.c_char * 0)()
    ret, mask = api.kdp_extract_feature_generic(dev_idx, data_p, constants.IMAGE_SIZE_RGB565_DEFAULT, 0, res)

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
    img_format_rgb565 = constants.IMAGE_FORMAT_SUB128 | constants.NPU_FORMAT_RGB565
    ret, img_size = api.kdp_start_sfid_mode(dev_idx, img_size, constants.FID_THRESHOLD, constants.IMAGE_SOURCE_W_DEFAULT, constants.IMAGE_SOURCE_H_DEFAULT, img_format_rgb565)

    if (ret != 0) or (img_size == 0):
        print("start verify mode")
        return -1

    sleep(constants.SLEEP_TIME)

    ret = api.kdp_remove_user(dev_idx, user_id)
    return ret

def start_inf_mode(dev_idx):
    img_size = 0
    img_format_rgb565 = constants.IMAGE_FORMAT_SUB128 | constants.NPU_FORMAT_RGB565
    ret, img_size = api.kdp_start_sfid_mode(dev_idx, img_size, constants.FID_THRESHOLD, constants.IMAGE_SOURCE_W_DEFAULT, constants.IMAGE_SOURCE_H_DEFAULT, img_format_rgb565)

    if (ret != 0) or (img_size == 0):
        print("start inf mode fail")
        return -1

    return ret

def verify_user_id(dev_idx, frame):
    mask = api.kdp_get_res_mask(1, 1, 0, 0)
    res_size = api.kdp_get_res_size(1, 1, 0, 0)
    res = (ctypes.c_char * res_size)()
    user_id = mask

    data_p = convert_numpy_to_char_p(frame, cv2.COLOR_BGR2BGR565)

    ret, u_id, mask_value = api.kdp_verify_user_id_generic(dev_idx, user_id, data_p, constants.IMAGE_SIZE_RGB565_DEFAULT, mask, res)

    fd_lm_res = ctypes.cast(ctypes.byref(res), ctypes.POINTER(constants.FDLMRes)).contents
    x = fd_lm_res.fd_res.x
    y = fd_lm_res.fd_res.y
    w = fd_lm_res.fd_res.w
    h = fd_lm_res.fd_res.h

    return u_id, x, y, w, h

def isi_get_result(dev_idx, img_id, r_data, app_id):
    """Gets inference results.

    Returns raw result data and result size on success. Returns None and -1 on failure. Raw
    result data will also be None if it is not returning the raw data.

    Arguments:
        dev_idx: Integer connected device ID
        img_id: Integer sequence ID to get inference results of an image with that ID
        r_data: Inference result data
        app_id: Integer ID of application to be run
    """
    rsp_code = ctypes.c_uint32(0)
    r_size = ctypes.c_uint32(0)
    ret = api.kdp_isi_retrieve_res(
        dev_idx, img_id, ctypes.byref(rsp_code), ctypes.byref(r_size), r_data)

    if ret:
        print(f"ISI get [{img_id}] result failed: {ret}")
        return None, -1

    if rsp_code.value:
        print(f"ISI get [{img_id}] result error_code: [{rsp_code.value}] [{r_size.value}]")
        return None, -1

    if r_size.value >= 4:
        if app_id == 0: # raw output
            # Prepare for postprocessing
            inf_res = cast_and_get(r_data, ctypes.c_char * r_size.value)
            listdata = [ord(byte) for byte in inf_res]
            npdata = np.asarray(listdata)

            fp_header_res = cast_and_get(inf_res, constants.RawFixpointData)
            output_num = fp_header_res.output_num

            outnode_params_res = cast_and_get(
                fp_header_res.out_node_params, constants.OutputNodeParams * output_num)

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
                offset = ctypes.sizeof(ctypes.c_int) + output_num * constants.OutputNodeParams().struct_size()
                # print("offset ", offset, ctypes.sizeof(c_int), ctypes.sizeof(OutputNodeParams))

                # get the fixed-point data
                npdata = npdata.astype("int8")

                raw_data = npdata[offset + data_offset:offset + data_offset + height * channel * pad_up_16(width)]
                data_offset += height * channel * pad_up_16(width)
                # print(raw_data.shape, offset, offset + height*channel*pad_up_16(width), height*channel*pad_up_16(width))
                raw_data = raw_data.reshape(height, channel, pad_up_16(width))
                raw_data = raw_data[:, :, :width]

                # save the fp data into numpy array and convert to float
                npraw_data = np.array(raw_data)
                npraw_data = npraw_data.transpose(0, 2, 1) / (2. ** radix) / scale
                npraw_data_array.append(npraw_data)

            return npraw_data_array, r_size.value
        elif app_id == constants.AppID.APP_AGE_GENDER: # age_gender
            gender = ["Female", "Male"]
            result = cast_and_get(r_data, constants.FDAgeGenderS)
            box_count = result.count
            print(f"Img [{img_id}]: {box_count} people")

            box = cast_and_get(result.boxes, constants.FDAgeGenderRes * box_count)

            for idx in range(box_count):
                print(f"[{idx}]: {gender[box[idx].ag_res.ismale]}, {box[idx].ag_res.age}")
        else: # od, yolo
            od_header_res = cast_and_get(r_data, constants.ObjectDetectionRes)
            box_count = od_header_res.box_count
            print(f"image {img_id} -> {box_count} object(s)")

        return None, r_size.value

    print(f"Img [{img_id}]: result_size {r_size.value} too small")
    return None, -1

def isi_capture_frame(cap, frames):
    """Frame read and convert to RGB565.

    Arguments:
        cap: Active cv2 video capture instance
        frames: List of frames for the video capture to add to
    """
    _cv_ret, frame = cap.read()

    if frame is None:
        print("fail to read from cam!")

    if sys.platform == "darwin":
        frame = cv2.resize(frame, (constants.IMAGE_SOURCE_W_DEFAULT, constants.IMAGE_SOURCE_H_DEFAULT),
                           interpolation=cv2.INTER_CUBIC)

    frame = cv2.flip(frame, 1)
    frames.append(frame)

    return convert_numpy_to_char_p(frame, cv2.COLOR_BGR2BGR565)

def setup_capture(cam_id, width, height):
    """Sets up the video capture device.

    Returns the video capture instance on success and None on failure.

    Arguments:
        cam_id: Integer camera ID
        width: Integer width of frames to capture
        height: Integer height of frames to capture
    """
    if sys.platform == "win32":
        capture = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    else:
        capture = cv2.VideoCapture(cam_id)
    if not capture.isOpened():
        print("Could not open video device!")
        return None
    if sys.platform != "darwin":
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return capture

def start_isi(device_index, app_id, width, height):
    """Starts the ISI mode.

    Returns 0 on success and -1 on failure.

    Arguments:
        device_index: Integer connected device ID
        app_id: Integer ID of application to be run
        width: Integer width of the input image
        height: Integer height of the input image
    """
    print("Starting ISI mode...")
    if app_id == constants.AppID.APP_OD:
        image_format = 0x80000060 | constants.IMAGE_FORMAT_CHANGE_ASPECT_RATIO
    else:
        image_format = 0x80000060
    size = constants.ISI_DETECTION_SIZE
    image_buf_size = ctypes.c_uint32(0)
    error_code = ctypes.c_uint32(0)

    ret = api.kdp_start_isi_mode(device_index, app_id, size, width, height, image_format,
                                 ctypes.byref(error_code), ctypes.byref(image_buf_size))
    if ret:
        print(f"Could not set to ISI mode: {ret}...")
        return -1
    if image_buf_size.value < 3:
        print(f"ISI mode window {image_buf_size.value} too small...")
        return -1

    print(f"ISI mode succeeded (window = {image_buf_size.value})...\n")
    sleep(constants.SLEEP_TIME)
    return 0

def start_isi_parallel_ext(device_index, app_id, width, height, is_raw_output=False):
    """Starts the ISI mode with isi configuration.

    Returns 0 on success and -1 on failure.

    Arguments:
        device_index: Integer connected device ID
        app_id: Integer ID of application to be run
        width: Integer width of the input image
        height: Integer height of the input image
        is_raw_output: Flag to get raw output data, false by default
    """
    print("Starting ISI mode...")
    # isi configuration
    extra_param = np.array([0.0])

    if app_id == constants.AppID.APP_OD:
        image_format = 0x88000060 | constants.IMAGE_FORMAT_CHANGE_ASPECT_RATIO
    else:
        image_format = 0x88000060

    if is_raw_output:
        image_format = image_format | constants.IMAGE_FORMAT_RAW_OUTPUT
        size = constants.ISI_RAW_SIZE_520
    else:
        size = constants.ISI_DETECTION_SIZE

    isi_cfg = constants.KDPISIConfig(app_id, size, width, height, image_format, extra_param)
    image_buf_size = start_isi_mode_ext(device_index, isi_cfg, isi_cfg.struct_size())

    if image_buf_size == -1:
        return -1

    print(f"ISI mode succeeded (window = {image_buf_size})...")
    sleep(constants.SLEEP_TIME)
    return 0

def isi_fill_buffer(device_index, capture, size, frames):
    """Fill up the image buffer using the capture device.

    Arguments:
        device_index: Integer onnected device ID
        capture: Active cv2 video capture instance
        size: Integer size of the input images
        frames: List of frames captured by the video capture instance
    """
    print("Starting ISI inference...")
    img_id_tx = 1234
    img_left = 12
    buffer_depth = 0
    while 1:
        data_p = isi_capture_frame(capture, frames)
        img_left = isi_inference(device_index, data_p, size, img_id_tx)
        if img_left == -1:
            print("Companion inference failed")
            return -1, img_id_tx, img_left, buffer_depth
        img_id_tx += 1
        buffer_depth += 1
        if not img_left:
            break
    return 0, img_id_tx, img_left, buffer_depth

def isi_pipeline_inference(device_index, app_id, loops, input_size, capture, img_id_tx,
                           img_left, buffer_depth, frames, post_handler, is_raw_output=False):
    """Send the rest of images and get the results.

    Returns 0 on success and -1 on failure.

    Arguments:
        device_index: Integer connected device ID
        app_id: ID of application to be run
        loops: Integer number of images to get results
        input_size: Integer size of the input images
        capture: Active cv2 video capture instance
        img_id_tx: Integer current image ID, should be returned from isi_fill_buffer()
        img_left: Integer number of images left, should be returned from isi_fill_buffer()
        buffer_depth: Integer number of images in buffer, should be returned from isi_fill_buffer()
        frames: List of frames captured by the video capture instance
        post_handler: Function to process the results of the inference
        is_raw_output: Flag to get raw output data, false by default
    """
    img_id_rx = 1234

    if is_raw_output:
        ret_size = constants.ISI_RAW_SIZE_520
    else:
        ret_size = constants.ISI_DETECTION_SIZE
    inf_res = (ctypes.c_char * ret_size)()
    while loops:
        if is_raw_output:
            app_id = 0 # for raw output
            npraw_data_array, result_size = isi_get_result(
                device_index, img_id_rx, inf_res, app_id)
            post_handler(device_index, npraw_data_array, frames)
        else:
            _, result_size = isi_get_result(device_index, img_id_rx, inf_res, app_id)
            post_handler(device_index, inf_res, result_size, frames)

        img_id_rx += 1
        data_p = isi_capture_frame(capture, frames)

        img_left = isi_inference(device_index, data_p, input_size, img_id_tx)
        if img_left == -1:
            return -1
        img_id_tx += 1
        loops -= 1

    # Get last 2 results
    while buffer_depth:
        if is_raw_output:
            app_id = 0 # for raw output
            npraw_data_array, result_size = isi_get_result(
                device_index, img_id_rx, inf_res, app_id)
            post_handler(device_index, npraw_data_array, frames)
        else:
            _, result_size = isi_get_result(device_index, img_id_rx, inf_res, app_id)
            post_handler(device_index, inf_res, result_size, frames)
        img_id_rx += 1
        buffer_depth -= 1
    return 0

def dme_fill_buffer(device_index, capture, input_size, frames, pre_handler=None):
    """Send 1 image to the DME image buffers using the capture device.

    Returns session ID.

    Arguments:
        device_index: Integer connected device ID
        capture: Active cv2 video capture instance
        input_size: Integer size of the input image
        frames: List of frames captured by the video capture instance
        pre_handler: Function to perform preprocessing; None uses capture, otherwise use frames
    """
    print("Starting DME inference...")
    inf_res = (ctypes.c_char * 0x300000)()
    res_flag = ctypes.c_bool(False)
    mode = 1
    model_id = 0
    ssid = ctypes.c_uint32(0)

    if capture is not None:
        img_buf = isi_capture_frame(capture, frames)
    elif pre_handler is not None:
        data = pre_handler(frames[0])
        img_buf = convert_numpy_to_char_p(data, size=input_size)
    else:
        print("Both capture and pre_handler function with input images in frames"
              " cannot be None...")
        return -1

    ret = api.kdp_dme_inference(
        device_index, img_buf, input_size, ctypes.byref(ssid), ctypes.byref(res_flag), inf_res, mode, model_id)
    if ret:
        print(f"DME inference failed: {ret}...")
        return -1

    return ssid.value

def dme_pipeline_inference(device_index, app_id, loops, input_size, capture,
                           prev_ssid, frames, post_handler, pre_handler=None):
    """Send the rest of images and get the results.

    Arguments:
        device_index: Integer connected device ID
        app_id: Integer ID of application to be run
        loops: Integer number of images to get results
        input_size: Integer size of the input image
        capture: Active cv2 video capture instance
        prev_ssid: Integer previous session ID, should be returned from dme_fill_buffer()
        frames: List of frames captured by the video capture instance
        post_handler: Function to process the results of the inference
        pre_handler: Function to perform preprocessing
    """
    inf_res = (ctypes.c_char * 0x300000)()
    res_flag = ctypes.c_bool(False)
    mode = 1
    model_id = 0
    ssid = ctypes.c_uint32(0)
    inf_size = ctypes.c_uint32(0)

    index = 1
    num_images = len(frames)
    while loops:
        prev_ssid = ctypes.c_uint16(prev_ssid)

        if capture is not None:
            img_buf = isi_capture_frame(capture, frames)
        elif pre_handler is not None:
            prev_index = (index - 1) % num_images
            cur_index = index % num_images
            data = pre_handler(frames[cur_index])
            img_buf = convert_numpy_to_char_p(data, size=input_size)
        else:
            print("Both capture and pre_handler function with input images in frames"
                  " cannot be None...")
            return -1

        ret = api.kdp_dme_inference(device_index, img_buf, input_size, ctypes.byref(ssid),
                                    ctypes.byref(res_flag), inf_res, mode, model_id)
        if ret:
            print(f"DME inference failed: {ret}...")
            return -1

        # get status for previous session
        while 1:
            status = ctypes.c_uint16(0)  # Must re-initialize status to 0
            ret = api.kdp_dme_get_status(device_index, ctypes.byref(prev_ssid),
                                         ctypes.byref(status), ctypes.byref(inf_size), inf_res)
            if ret:
                print(f"Get DME status failed: {ret}...")
                return -1

            if status.value == 1:
                npraw_data = dme_get_result(device_index, inf_size.value, app_id)
                if capture is not None:
                    post_handler(device_index, npraw_data, frames)
                else:
                    post_handler(device_index, npraw_data, frames[prev_index])
                break

        prev_ssid = ssid.value
        loops -= 1
        index += 1

    # Get last 1 results
    prev_ssid = ctypes.c_uint16(prev_ssid)
    while 1:
        status = ctypes.c_uint16(0)  # Must re-initialize status to 0
        ret = api.kdp_dme_get_status(device_index, ctypes.byref(prev_ssid),
                                     ctypes.byref(status), ctypes.byref(inf_size), inf_res)
        if ret:
                print(f"Get DME status failed: {ret}...")
                return -1

        if status.value == 1:
            npraw_data = dme_get_result(device_index, inf_size.value, app_id)
            if capture is not None:
                post_handler(device_index, npraw_data, frames)
            else:
                prev_index = (index - 1) % num_images
                post_handler(device_index, npraw_data, frames[prev_index])
            break

    return 0

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

    img_left = isi_inference(dev_idx, buffer, size, img_id_tx)
    if img_left == -1:
        return -1
    img_id_tx += 1

    img_left = isi_inference(dev_idx, buffer_t, size, img_id_tx)
    if img_left == -1:
        return -1
    img_id_tx += 1

    return img_left

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
        img_left = isi_inference(dev_idx, buffer, input_size, img_id_tx)
        if img_left == -1:
            return -1
        img_id_tx += 1

        _, result_size = isi_get_result(dev_idx, img_id_rx, inf_res, app_id)
        img_id_rx += 1

        loop -= 1
        # Odd loop case
        if not loop:
            break

        img_left = isi_inference(dev_idx, buffer_t, input_size, img_id_tx)
        if img_left == -1:
            return -1
        img_id_tx += 1

        _, result_size = isi_get_result(dev_idx, img_id_rx, inf_res, app_id)
        img_id_rx += 1
        loop -= 1

    return img_id_rx

def isi_get_last_results(dev_idx, app_id, img_id_rx, ret_size):
    """Gets results for last two images.

    Arguments:
        dev_idx: Connected device ID. A host can connect several devices.
        app_id: ID of application to be run.
        img_id_rx: Sequence ID to get inference results of an image with that ID
        ret_size: Return size.
    """
    inf_res = (ctypes.c_char * ret_size)()
    _, result_size = isi_get_result(dev_idx, img_id_rx, inf_res, app_id)
    img_id_rx += 1

    _, result_size = isi_get_result(dev_idx, img_id_rx, inf_res, app_id)
    img_id_rx += 1

    return 0

def get_kn_number(device_index):
    """Request for device KN number.

    Returns the KN number on success, -1 on failure, and -2 on no KN number available.

    Arguments:
        device_index: Integer connected device ID
    """
    kn_num = ctypes.c_uint32(0)
    ret = api.kdp_get_kn_number(device_index, ctypes.byref(kn_num))

    if ret:
        print("Could not get KN number...")
        return -1

    if kn_num.value == 0xFFFF:
        print("Not supported by the version of the firmware")
        return -2
    return kn_num.value

def get_model_info(device_index, from_ddr):
    """Request for model IDs in DDR or Flash.

    Returns list of models on success and empty list on failure.

    Arguments:
        device_index: Integer connected device ID
        from_ddr: if models are in ddr (1) or flash (0)
    """
    r_data = (ctypes.c_char * 1024)()
    # Data: total_number (4 bytes) + model_id_1 (4 bytes) + model_id_2 (4 bytes) + ...
    ret = api.kdp_get_model_info(device_index, from_ddr, r_data)

    if ret:
        print("Could not get model info...")
        return []

    modelinfo = ctypes.cast(ctypes.byref(r_data), ctypes.POINTER(ctypes.c_int * 256)).contents
    return modelinfo[:modelinfo[0] + 1]

def get_crc(device_index, from_ddr):
    """Request for CRC in DDR or Flash.

    Returns CRC on success, -1 on failure, and -2 on no CRC info available.

    Arguments:
        device_index: Integer connected device ID
        from_ddr: if models are in ddr (1) or flash (0)
    """
    r_data = (ctypes.c_char * 4)()
    # Get the data for crc (4 bytes)
    ret = api.kdp_get_crc(device_index, from_ddr, r_data)

    if ret:
        print("Could not get CRC...")
        return -1

    crcinfo = ctypes.cast(ctypes.byref(r_data), ctypes.POINTER(ctypes.c_uint * 256)).contents
    crc = crcinfo[0]

    if crc == 0xFFFF:
        print("Not supported by the version of the firmware\n")
        return -2
    elif crc == 0 and from_ddr:
        print("Models have not been loaded into DDR\n")
        return -2
    elif crc == 0xFFFFFFFF:
        print("No CRC info for the loaded models\n")
        return -2
    return crc

def get_nef_model_metadata(model_path):
    """Request for metadata from NEF model file.

    Returns the KDPNEFMetadata instance on success and None on failure.

    Arguments:
         model_path: NEF model file
    """
    metadata = constants.KDPNEFMetadata()

    p_buf, model_size = read_file_to_buf_with_size(model_path, constants.MAX_MODEL_SIZE_720)

    ret = api.kdp_get_nef_model_metadata(p_buf, model_size, ctypes.byref(metadata))

    if ret:
        print("Could not get metadata...")
        return None

    return metadata

def reset_sys(device_index, reset_mode):
    """System reset.

    Returns 0 on success and error code on failure.

    Arguments:
        device_index: Integer connected device ID
        reset_mode: Integer mode
            0 - no operation
            1 - reset message protocol
            3 - switch to suspend mode
            4 - switch to active mode
            255 - reset whole system
            256 - system shutdown (RTC)
            0x1000xxxx - reset debug output level
    """
    ret = api.kdp_reset_sys(device_index, reset_mode)

    if ret:
        print("Could not reset sys...")
    else:
        print("Sys reset mode succeeded...")

    return ret

def start_isi_mode_ext(device_index, isi_config, cfg_size):
    """Start the user ISI mode with the ISI configuration data.

    Returns image buffer size on success and -1 on failure.

    Arguments:
        device_index: Integer connected device ID
        isi_config: KDPISIConfig instance
        cfg_size: Integer size of KDPISIConfig class
    """
    error_code = ctypes.c_uint32(0)
    buf_size = ctypes.c_uint32(0)
    isi_config_p = ctypes.cast(ctypes.pointer(isi_config), ctypes.POINTER(ctypes.c_char))

    ret = api.kdp_start_isi_mode_ext(device_index, isi_config_p, cfg_size,
                                     ctypes.byref(error_code), ctypes.byref(buf_size))

    if ret:
        print(f"Could not set to ISI mode: {ret}...")
        return -1
    if buf_size.value < 3:
        print(f"ISI mode window {buf_size.value} is too small...")
        return -1

    return buf_size.value

def start_isi_mode_ext2(device_index, app_id, compatible_cfg, isi_data, p_buf, model_size):
    """Initializes data and calls kdp_start_isi_mode_ext2() Python ctypes function wrapper.

    Returns 0 on success and error code on failure. Will also return image buffer size.

    Arguments:
        device_index: Integer connected device ID
        app_id: Integer application ID to run
        compatible_cfg: TODO
        isi_data: KAppISIData instance, may be None
        p_buf: Model buffer, should be result of read_file_to_buf_with_size()
        model_size: Integer model buffer size, should be result of read_file_to_buf_with_size()
    """
    dummy_init = (ctypes.c_char * constants.ISI_START_DATA_SIZE_720)()
    isi_init = ctypes.cast(dummy_init, ctypes.POINTER(constants.KDPISIStart))

    # set isi start fields
    isi_init.contents.app_id = app_id
    isi_init.contents.compatible_cfg = compatible_cfg
    if isi_data is not None:
        isi_init.contents.start_flag = constants.DOWNLOAD_MODEL | constants.DEFAULT_CONFIG
        isi_init.contents.config = isi_data
        # should be 208: 28 (7 int fields) + 180 (1 KAppISIModelConfig)
        config_size = constants.KDPISIStart().struct_size()
    else:
        isi_init.contents.start_flag = constants.DOWNLOAD_MODEL
        # should be 12 (3 int fields)
        config_size = constants.KDPISIStart().struct_size() - constants.KAppISIData().struct_size()

    error_code = ctypes.c_uint32(0)
    buf_size = ctypes.c_uint32(0)

    ret = api.kdp_start_isi_mode_ext2(device_index, dummy_init, config_size, p_buf, model_size,
                                      ctypes.byref(error_code), ctypes.byref(buf_size))
    if ret:
        print(f"kdp_start_isi_ext2 failed: {ret} (model_size {model_size}, ret = {ret})")
    return ret

def start_det_isi_mode_ext2(device_index, p_buf, model_size, app_id, compatible_cfg=0):
    """Initializes data and calls kdp_start_isi_mode_ext2() Python ctypes function wrapper.

    Returns 0 on success and error code on failure.

    Arguments:
        device_index: Integer connected device ID
        p_buf: Model buffer, should be result of read_file_to_buf_with_size()
        model_size: Integer model buffer size, should be result of read_file_to_buf_with_size()
        app_id: Integer application ID
        compatible_cfg: TODO
    """

    if app_id == constants.AppID.APP_PDC.value:
        image_format = (constants.IMAGE_FORMAT_SUB128 |
                        constants.NPU_FORMAT_RGB565)

        # original params
        # POST_PROC_PARAMS_V5S = [
        #     0.45, 0.45, 20, 3, 6, 3, 0, 10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116,
        #     90, 156, 198, 373, 326, 8, 16, 32]
        # follow the structure of ODPostParameter720
        POST_PROC_PARAMS_V5S = [
            0.45, 0.45, 20, 0x00060003, 0x00000003, 10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116,
            90, 156, 198, 373, 326, 8, 16, 32]

        # convert float to int data type
        for i in range(len(POST_PROC_PARAMS_V5S)):
            if i >= 2:
                continue

            scale = struct.unpack("I", struct.pack("f", float(POST_PROC_PARAMS_V5S[i])))[0]
            POST_PROC_PARAMS_V5S[i] = scale

        # TODO may need to check 2048 output size
        # isi_data = constants.KAppISIData(
        #     size=size, version=0, output_size=2048, config_block=1, m=[model_config])
        isi_data = init_isi_data(3, 640, 480, image_format, 0, POST_PROC_PARAMS_V5S)
    else:
        print("Add configuration for appid: ", app_id)
        return -1

    return start_isi_mode_ext2(device_index, app_id, compatible_cfg, isi_data, p_buf, model_size)

def isi_inference(device_index, input_image, buf_len, img_id, frames=[]):
    """Start an inference with an image or from a video capture.

    Returns number of available images left on success and -1 on failure.

    Arguments:
        device_index: Integer connected device ID
        input_image: Image buffer from result of read_file_to_buf_with_size() or active cv2
            video capture instance
        buf_len: Integer image buffer size
        img_id: Integer ID of the test image
        frames: List of frames for the video capture to add to, only used if input_image is
            cv2 capture
    """
    error_code = ctypes.c_uint32(0)
    img_buf_left = ctypes.c_uint32(0)

    if isinstance(input_image, cv2.VideoCapture):
        img_buf = isi_capture_frame(input_image, frames)
    else:
        img_buf = input_image

    ret = api.kdp_isi_inference(device_index, img_buf, buf_len, img_id,
                                ctypes.byref(error_code), ctypes.byref(img_buf_left))
    if ret:
        print(f"ISI inference failed: {ret}")
        return -1

    if error_code.value:
        print(f"ISI inference error_code: [{error_code.value}] [{img_buf_left.value}]")
        return -1
    return img_buf_left.value

def isi_inference_ext(device_index, img_header, capture, buf_len, img_id, frames):
    """Start an inference with an image from a video capture.

    Returns 0 on success and error code on failure. Will also return number of available
    images left.

    Arguments:
        device_index: Integer connected device ID
        capture: Active cv2 video capture instance
        buf_len: Integer image buffer size
        img_id: Integer ID of the test image
        frames: List of frames for the video capture to add to
    """
    error_code = ctypes.c_uint32(0)
    img_buf_left = ctypes.c_uint32(0)
    data_p = isi_capture_frame(capture, frames)
    header_len = 32
    # def kdp_isi_inference_ext(device_index, img_buf, img_header,
    #                           header_len, rsp_code, img_buf_available):
    ret = api.kdp_isi_inference_ext(device_index, data_p, img_header, header_len,
                                ctypes.byref(error_code), ctypes.byref(img_buf_left))
    return ret, error_code.value, img_buf_left.value

def isi_retrieve_res(device_index, img_id):
    """Retrieves inference results.

    Returns result data buffer and result size on success and None and -1 on failure.

    Arguments:
        device_index: Integer connected device ID
        img_id: Integer image ID to get inference results for
    """
    error_code = ctypes.c_uint32(0)
    result_size = ctypes.c_uint32(0)
    result_data = (ctypes.c_char * constants.ISI_RESULT_SIZE_720)()

    ret = api.kdp_isi_retrieve_res(device_index, img_id, ctypes.byref(error_code),
                                   ctypes.byref(result_size), result_data)

    if ret:
        print(f"ISI get {img_id} result failed: {ret}")
        return None, -1

    if error_code.value:
        print(f"ISI get {img_id} result error_code: {error_code.value} {result_size.value}")
        return None, -1

    if result_size.value < 4: # 4 = sizeof(uint32_t)
        print(f"Image {img_id}: result_size {result_size.value} too small")
        return None, -1

    return result_data, result_size.value

def isi_config(device_index, model_id, param):
    """Configure the model for the supported model ID.

    Returns 0 on success and -1 on failure.

    Arguments:
        device_index: Integer connected device ID
        model_id: Integer model ID to run inference
        param: Integer param needed for the model
    """
    error_code = ctypes.c_uint32(0)
    ret = api.kdp_isi_config(device_index, model_id, param, ctypes.byref(error_code))

    if ret or error_code.value:
        print(f"Could not configure to model {model_id}: {ret}, {error_code.value}")
        return -1
    return 0

def read_file_to_buf(image_file, image_size):
    """Reads input image into a buffer.

    Returns the image buffer and None on failure.

    Arguments:
        image_file: String path to the input image
        image_size: Integer size of the input image
    """
    buffer = (ctypes.c_char * image_size)()
    length = api.read_file_to_buf(buffer, image_file.encode(), image_size)
    if length <= 0:
        print(f"Reading file {image_file} failed: {length}...")
        return None
    return buffer

def read_file_to_buf_with_size(image_file, image_size):
    """Reads input image into a buffer.

    Returns the image buffer and length of the input image. Imabe buffer will be none,
    and length will be negative on failure.

    Arguments:
        image_file: String path to the input image
        image_size: Integer size of the input image
    """
    buffer = (ctypes.c_char * image_size)()
    length = api.read_file_to_buf(buffer, image_file.encode(), image_size)
    if length <= 0:
        print(f"Reading file {image_file} failed: {length}...")
        return None, length
    return buffer, length

def cast(data, result_class):
    """Cast the data pointer to the provided class type.

    Return the new data pointer of new type.

    Arguments:
        data: Ctypes pointer to data
        result_class: Class to cast the data pointer to
    """
    return ctypes.cast(data, ctypes.POINTER(result_class))

def cast_and_get(data, result_class):
    """Cast the data pointer to the provided class type and get the data.

    Return the data after casting the pointer.

    Arguments:
        data: Ctypes pointer to data
        result_class: Class to cast the data pointer to
    """
    return ctypes.cast(data, ctypes.POINTER(result_class)).contents

def isi_load_nef(device_index, nef_file, app_id, compatible_cfg=0, isi_data=None):
    """Loads NEF file and starts ISI mode.

    Returns 0 on success and error code on failure.

    Arguments:
        device_index: Integer connected device ID
        nef_file: String path to NEF model
        app_id: Integer application ID to run
        compatible_cfg: TODO
        isi_data: KAppISIData instance
    """
    p_buf, model_size = read_file_to_buf_with_size(nef_file, constants.MAX_MODEL_SIZE_720)
    if model_size <= 0:
        return model_size

    ret = start_isi_mode_ext2(device_index, app_id, compatible_cfg, isi_data, p_buf, model_size)
    if ret:
        print(f"Failed to load NEF file, {nef_file}, onto the board...")
    return ret

def convert_data_to_numpy(result_data, add_batch=False, channel_last=False):
    """Converts raw output data into list of NumPy arrays.

    Returns a list of NumPy arrays.

    Arguments:
        result_data: Output data, should be result of isi_retrieve_res()
        add_batch: Flag to add batch dimension
        channel_last: Flag to set output results in channel last format
    """
    imga = np.frombuffer(result_data, np.int8)  # int8
    imgb = np.frombuffer(result_data, np.int32)
    imgb_float = np.frombuffer(result_data, np.float)
    imgc = np.frombuffer(result_data, np.int16)  # int16
    # print("raw len", imgb[0])
    outnum = imgb[1]
    offset = 0 # size for read data
    results = []
    for i in range(outnum):
        #parse header
        h = imgb[14 * i + 10]
        w = imgb[14 * i + 11]
        c = imgb[14 * i + 12]
        radix = imgb[14 * i + 14]
        scale = imgb[14 * i + 15]

        # convert "int" scale value into float value using its bytes
        scale = struct.unpack("f", struct.pack("I", int(scale)))[0]
        # print(1.0/scale)
        # print(radix, scale)

        # for j in range(14):
        #     print(imgb[14 * i + 2 + j])

        size = imgb[14 * i + 6] + 1
        # print("data size", size)

        if w % 16 == 0:
            wnew = w
        else:
            wnew = (w // 16 + 1) * 16

        #in 1 byte: 40 is total node for latest fw, 14 is struct for each node, 2 is
        start = (40 * 14 + 2) * 4 + offset * size
        end = start + h * c * wnew * size
        offset += h * c * wnew * size

        if size == 1:
            data = imga[start:end]
        if size == 2:
            data = imgc[start//2:end//2]
        # print(data)
        #chw
        data = data.reshape(c, h, wnew)
        data = data[:, :, :w] # remove aligned bytes

        if channel_last:    # set to h, w, c if specified
            data = np.transpose(data, (1, 2, 0))

        if add_batch:   # add batch dimension
            data = data.reshape(1, *data.shape)

        # data = data.reshape(-1, 1)
        data = data.astype(np.float64) / scale / (2.0 ** radix)

        results.append(data)
    return results

def scan_usb_devices():
    """Scan all Kneron devices and report a list.

    Returns 0. Will also return list of all available devices.
    """
    # KDPDeviceInfoList instance
    dev_info_list = constants.KDPDeviceInfoList()

    # Create double pointer to KDPDeviceInfoList
    r_data = ctypes.pointer(ctypes.pointer(dev_info_list))
    ret = api.kdp_scan_usb_devices(r_data)

    # Get the device info list
    dev_info_list = []

    dev_list = r_data.contents.contents
    for i in range(dev_list.num_dev):
        dev_info = []
        dev_info.append(dev_list.kdevice[i].scan_index)
        dev_info.append(dev_list.kdevice[i].isConnectable)
        dev_info.append(dev_list.kdevice[i].vendor_id)
        dev_info.append(dev_list.kdevice[i].product_id)
        dev_info.append(dev_list.kdevice[i].link_speed)
        dev_info.append(dev_list.kdevice[i].serial_number)
        dev_info.append(dev_list.kdevice[i].device_path.decode())
        dev_info_list.append(dev_info)

    # print("return list:", dev_info_list)
    return ret, dev_info_list

# ISI center app helpers
def print_detail_results(img_id, yolo_result):
    """Print results of the inference.

    Arguments:
        img_id: Integer ID of test image
        yolo_result: YoloResult instance
    """
    print(f"Image {img_id} result details:")
    print(f"- class count: {yolo_result.class_count}")
    print(f"- object count: {yolo_result.box_count}")

    boxes = []
    for index, box in enumerate(yolo_result.boxes[:yolo_result.box_count]):
        box_as_list = [box.x1, box.y1, box.x2, box.y2, box.score, box.class_num]
        boxes.append(box_as_list)

        print(f"- box {index}: ({box.x1:.0f}, {box.y1:.0f}) ({box.x2:.0f}, {box.y2:.0f}) score "
              f"{box.score:.3f} classnum {box.class_num}")

    global g_results
    g_results = [boxes]

def check_result(img_id, result_size, result_data, s_result_size,
                 s_result_hex, skip_content_errors):
    """Check results of inference.

    Returns True if results are good and False otherwise. Will also return number of content
    errors if not ignored.

    Arguments:
        img_id: Integer ID of test image
        result_size: Integer size of result buffer, should be result of isi_retrieve_res
        result_data: Ctypes char array holding the results, should be result of isi_retrieve_res
        s_result_size: List of saved box counts
        s_result_hex: List of saved box contents
        skip_content_errors: Flag to skip the content errors
    """
    yolo_result = cast_and_get(result_data, constants.YoloResult)
    result_compare_size = False
    result_compare_content = False
    if not skip_content_errors:
        if img_id <= 2:
            print_detail_results(img_id, yolo_result)

            s_result_size[img_id - 1] = result_size
            s_result_hex[img_id - 1] = yolo_result

            return True, 0

        box_count = yolo_result.box_count
        print(f"Image {img_id} -> {box_count} object(s)")

        if result_size == s_result_size[1 - (img_id % 2)]:
            result_compare_size = True
        if yolo_result == s_result_hex[1 - (img_id % 2)]:
            result_compare_content = True

        if result_compare_size:
            print("... size correct")
        else:
            print("... size failed")

        if result_compare_content:
            print("... content correct")
        else:
            print("... content failed")

    if (not result_compare_size or not result_compare_content) and skip_content_errors:
        print_detail_results(img_id, yolo_result)
        return True, 1

    return result_compare_size and result_compare_content, 0

def get_detection_res_center_app(model_id, img_id, result_size, result_data,
                                 s_result_size, s_result_hex, skip_content_errors):
    """Get the detection results.

    Arguments:
        model_id: Integer ID of the test model
        img_id: Integer ID of the test image
        result_size: Integer size of result buffer, should be result of isi_retrieve_res()
        result_data: Ctypes char array holding the results, should be result of isi_retrieve_res()
        s_result_size: List of saved box counts
        s_result_hex: List of saved box contents
        skip_content_errors: Flag to skip the content errors
    """
    model_set = {
        constants.ModelType.KNERON_OBJECTDETECTION_CENTERNET_512_512_3.value,
        constants.ModelType.YOLO_V4_416_416_3.value,
        constants.ModelType.KNERON_FD_MBSSD_200_200_3.value,
        constants.ModelType.TINY_YOLO_V3_416_416_3.value,
        constants.ModelType.TINY_YOLO_V3_608_608_3.value,
        constants.ModelType.YOLO_V3_416_416_3.value, constants.ModelType.YOLO_V3_608_608_3.value,
        constants.ModelType.KNERON_YOLOV5S_640_640_3.value,
        constants.ModelType.KNERON_YOLOV5M_640_640_3.value,
        constants.ModelType.CUSTOMER_MODEL_1.value, constants.ModelType.CUSTOMER_MODEL_2.value,
        constants.ModelType.CUSTOMER_MODEL_3.value, constants.ModelType.CUSTOMER_MODEL_4.value,
        constants.ModelType.KNERON_PERSONDETECTION_YOLOV5s_480_256_3.value,
        constants.ModelType.KNERON_PERSONDETECTION_YOLOV5sParklot_480_256_3.value}

    if model_id in model_set:
        ret, content_errors = check_result(
            img_id, result_size, result_data, s_result_size, s_result_hex, skip_content_errors)
        if not ret:
            return -1, g_results, content_errors
    elif (model_id == constants.ModelType.IMAGENET_CLASSIFICATION_MOBILENET_V2_224_224_3.value or
          model_id == constants.ModelType.KNERON_PERSONCLASSIFIER_MB_56_32_3.value):
        imgnet_res = cast_and_get(result_data, constants.ImageNetResult)
        print(imgnet_res)
    elif model_id == constants.ModelType.KNERON_LM_5PTS_ONET_56_56_3.value:
        lm_res = cast_and_get(result_data, constants.LandmarkResult)
        print(lm_res)
    elif model_id == constants.ModelType.KNERON_FR_VGG10.value:
        fr_res = cast_and_get(result_data, constants.FRResult)
        print(fr_res)
    else:
        print(f"Wrong model ID {model_id} with size {result_size}")

    return 0, g_results, 0

# end ISI center app helpers

def init_log(directory, name):
    """Initialize the host lib internal log.

    Returns 0 on success and -1 on failure.

    Arguments:
        directory: String name of directory
        name: String name of log file
    """
    return api.kdp_init_log(directory.encode(), name.encode())

def update_fw(device_index, module_id, fw_file, fw_size):
    """Update firmware.

    Returns 0 on success and error code on failure.

    Arguments:
        device_index: Integer connected device ID
        module_id: Integer module ID of which firmware to be updated
            0 - no operation
            1 - SCPU module
            2 - NCPU module
        fw_file: String path to firmware data
        fw_size: Integer size of firmware data
    """
    img_buf, img_buf_size = read_file_to_buf_with_size(fw_file, fw_size)
    if img_buf_size <= 0:
        print(f"Reading model file, {fw_file}, failed: {img_buf_size}")
        return img_buf_size

    module = ctypes.c_uint32(module_id)
    ret = api.kdp_update_fw(device_index, ctypes.byref(module), img_buf, img_buf_size)

    module = {0: "nothing", 1: "SCPU", 2: "NCPU"}
    module_string = module.get(module_id, "UNKNOWN MODULE")
    if ret:
        print(f"Could not update {module_string}...")
    else:
        print(f"Update {module_string} firmware succeeded...")

    return ret

def report_sys_status(device_index):
    """Reports device status.

    Returns 0 on success and error code on failure.

    Arguments:
        device_index: Integer connected device ID
    """
    sfw_id = ctypes.c_uint32(0)
    sbuild_id = ctypes.c_uint32(0)
    sys_status = ctypes.c_uint16(0)
    app_status = ctypes.c_uint16(0)
    nfw_id = ctypes.c_uint32(0)
    nbuild_id = ctypes.c_uint32(0)

    ret = api.kdp_report_sys_status(
        device_index, ctypes.byref(sfw_id), ctypes.byref(sbuild_id), ctypes.byref(sys_status),
        ctypes.byref(app_status), ctypes.byref(nfw_id), ctypes.byref(nbuild_id))

    if ret:
        print("Could not report sys status...")
        return ret
 
    print("Report sys status succeeded...")
    architecture = sfw_id.value >> 24
    major = (sfw_id.value & 0x00ff0000) >> 16
    minor = (sfw_id.value & 0x0000ff00) >> 8
    update = (sfw_id.value & 0x000000ff)
    build = sbuild_id.value

    print(f"\nFW firmware_id {architecture}.{major}.{minor}.{update}, build_id {build}\n")

    return ret

def update_nef_model(device_index, nef_file, nef_size):
    """Updates NEF model.

    Returns 0 on success and error code on failure.

    Arguments:
        device_index: Integer connected device ID
        nef_file: String path to NEF model
        nef_size: Integer size of NEF model
    """
    p_buf, model_size = read_file_to_buf_with_size(nef_file, nef_size)
    if model_size <= 0:
        print(f"Reading model file, {nef_file}, failed: {model_size}")
        return model_size

    ret = api.kdp_update_nef_model(device_index, p_buf, model_size)
    if ret:
        print("Could not update model...")
    else:
        print("Update model succeeded...")

    return ret

def lib_init():
    """Initialize the host library.

    Returns 0 on success and -1 on failure.
    """
    return api.kdp_lib_init()

def lib_start():
    """Start the host library to wait for messages.

    Returns 0 on success and -1 on failure.
    """
    return api.kdp_lib_start()

def lib_de_init():
    """Free the resources used by host lib.

    Returns 0 on success and -1 on failure.
    """
    return api.kdp_lib_de_init()

def connect_usb_device(scan_index=1):
    """Connect to a Kneron device via 'scan_index'.

    Returns device index on success and negative value on failure.

    Arguments:
        scan_index: Integer device index to connect
    """
    return api.kdp_connect_usb_device(scan_index)

def init_isi_data(image_channel, image_col, image_row, image_format, model_id, post_proc_params=[0]):
    """Initialize ISI data instance for configuration.

    Returns initialized KAppISIData instance.

    Arguments:
        image_channel: Integer number of channels in input image
        image_col: Integer width of input image
        image_row: Integer height of input image
        image_format: Integer format of input image
        model_id: Integer model ID to be inferenced
        post_proc_params: List of postprocess parameters
    """
    image_info = constants.KDPImgDesc(image_col, image_row, image_channel, image_format)
    model_config = constants.KAppISIModelConfig(model_id, image_info, post_proc_params)

    isi_data = constants.KAppISIData(version=0, output_size=constants.ISI_RESULT_SIZE_720,
                                     config_block=1, m=[model_config])
    # size should be 16 (4 int fields) + size of KAppISIModelConfig * number of model configs
    isi_data.size = isi_data.struct_size()

    return isi_data

def load_rgb565_bin_image(input_file, src_w, src_h):
    """Loads binary RGB565 image as RGB.

    Arguments:
        input_file: String path to binary image
        src_w: Integer width of image
        src_h: Integer height of image
    """
    # load bin
    struct_fmt = '1B'
    struct_len = struct.calcsize(struct_fmt)
    struct_unpack = struct.Struct(struct_fmt).unpack_from

    row = src_h
    col = src_w
    pixels = row*col

    rgba565 = []
    with open(input_file, "rb") as f:
        while True:
            data = f.read(struct_len)
            if not data: break
            s = struct_unpack(data)
            rgba565.append(s[0])

    rgba565 = rgba565[:pixels*2]

    # rgb565_bin to numpy_array
    output = np.zeros((pixels * 3), dtype=np.uint8)
    cnt = 0
    for i in range(0, pixels*2, 2):
        temp = rgba565[i]    ##Byte 0
        temp2 = rgba565[i+1] ##Byte 1
        #R-5
        output[cnt] = (temp2 >>3)<< 3

        #G-6
        cnt += 1
        output[cnt] = ((temp & 0xe0) >> 5) + ((temp2 & 0x07) << 3)<< 2

        #B-5
        cnt += 1
        output[cnt] = (temp & 0x1f)<< 3

        cnt += 1

    output = output.reshape((src_h,src_w,3))
    return output

def display(image, dets, class_path, det_type="xywh", image_format=None, size=None):
    """Displays detection results.

    Arguments:
        image: String path to image
        dets: List of bounding boxes
        class_path: String path to classes file
        det_type: String indicating if boxes are in 'xywh' or 'x1y1x2y2' format
        image_format: String image format if using binary file
        size: Tuple of width by height if using binary file
    """
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    if image.endswith(".bin"):
        if image_format.lower() == "rgb565":
            img = load_rgb565_bin_image(image, size[0], size[1])

    with open(class_path) as class_file:
        classes = class_file.readlines()
    classes = [c.strip() for c in classes]

    fig = plt.figure(figsize=(15, 15))
    plot = fig.add_subplot(111)
    plot.imshow(img)

    for box in dets:
        if det_type == "xywh":
            rect = patches.Rectangle(
                (box[0], box[1]), box[2], box[3],
                 linewidth=1, edgecolor='r', fill=False)
        elif det_type == "xyxy":
            rect = patches.Rectangle(
                (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                 linewidth=1, edgecolor='r', fill=False)
        plot.add_patch(rect)

        class_name = classes[int(box[5])] if classes is not None else ""
        label = f"{class_name} {box[4]:.2f}"
        plot.text(box[0], box[1], label, bbox=dict(facecolor='red', alpha=0.5))

    plt.show()
    plt.close()

def overlap(x1_1, y1_1, x2_1, y2_1, x1_2, y1_2, x2_2, y2_2):
    """Determine amount of overlap between the coordinates."""
    #check if there is any overlap
    if x1_1 > x2_2 or x2_1 < x1_2:
        return 0
    elif y1_1 > y2_2 or y2_1 < y1_2:
        return 0

    #if there is overlap
    if x1_1 > x1_2:
        x1 = x1_1
    else:
        x1 = x1_2

    if y1_1 > y1_2:
        y1 = y1_1
    else:
        y1 = y1_2

    if x2_1 < x2_2:
        x2 = x2_1
    else:
        x2 = x2_2

    if y2_1 < y2_2:
        y2 = y2_1
    else:
        y2 = y2_2

    w1 = x2_1 - x1_1
    h1 = y2_1 - y1_1
    w = x2 - x1
    h = y2 - y1
    print(w, h, w1, h1)
    o_l = float(w * h) / float(w1 * h1)
    return o_l

def draw_capture_result(device_index, dets, frames, det_type,
                        xywh=False, apple_img=None, apple_list=None):
    """Draw the detection results on the given frames.

    Arguments:
        device_index: Integer connected device ID
        dets: List of detection results
        frames: List of frames, result will be drawn on first frame
        det_type: String indicating which model result dets corresponds to
        xywh: Flag indicating if dets are in xywh, default is x1y1x2y2
        apple_img: Apple image array to display
        apple_list: List of apples, only used if det_type is 'apple'
    """
    x1_0 = 0
    y1_0 = 0
    x2_0 = 0
    y2_0 = 0
    # score_0 = 0

    # for multiple faces
    for det in dets:
        x1 = det[0]
        y1 = det[1]
        x2 = det[2]
        y2 = det[3]

        if xywh:    # convert xywh to x1y1x2y2
            x2 += x1
            y2 += y1

        if det_type == "age_gender":
            score = det[4]
            age = det[5]
            gender = det[6]
            if gender == 0:
                frames[0] = cv2.rectangle(frames[0], (x1, y1), (x2, y2), (0, 0, 255), 3)
                frames[0] = cv2.putText(frames[0], str(age), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (0, 0, 255), 2, cv2.LINE_AA)
            elif gender == 1:
                frames[0] = cv2.rectangle(frames[0], (x1, y1), (x2, y2), (255, 0, 0), 3)
                frames[0] = cv2.putText(frames[0], str(age), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (255, 0, 0), 2, cv2.LINE_AA)
        elif det_type == "yolo" or det_type == "fd_no_overlap" or det_type == "fd_mask_no_overlap":
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            score = det[4]
            class_num = det[5]

            if det_type == "fd_mask_no_overlap":
                if class_num == 2:
                    frames[0] = cv2.rectangle(frames[0], (x1, y1), (x2, y2), (0, 0, 255), 3)
                elif class_num == 1:
                    frames[0] = cv2.rectangle(frames[0], (x1, y1), (x2, y2), (255, 0, 0), 3)
            else:
                if class_num == 0:
                    frames[0] = cv2.rectangle(frames[0], (x1, y1), (x2, y2), (0, 0, 255), 3)
                else:
                    frames[0] = cv2.rectangle(frames[0], (x1, y1), (x2, y2), (255, 0, 0), 3)
        else:
            class_num = det[4]
            score = det[5]
            o_l = overlap(x1, y1, x2, y2, x1_0, y1_0, x2_0, y2_0)
            if o_l < 0.6:
                x1_0 = x1
                y1_0 = y1
                x2_0 = x2
                y2_0 = y2
                # score_0 = score
                if class_num == 2:
                    frames[0] = cv2.rectangle(frames[0], (x1, y1), (x2, y2), (0, 0, 255), 3)
                    #print("score of mask fd: ", score)
                elif class_num == 1:
                    frames[0] = cv2.rectangle(frames[0], (x1, y1), (x2, y2), (255, 0, 0), 3)
                    #print("score of fd: ", score)

                if det_type == "apple":
                    index = 0
                    remove_index = -1

                    # only remove one per frame
                    for apple in apple_list:
                        apple_x = apple[0] + 50
                        apple_y = apple[1] + 50

                        if x1 < apple_x < x2 and (y1 + (y2 - y1) / 2) < apple_y < y2:
                            remove_index = index
                        index += 1

                    if remove_index != -1:
                        del apple_list[remove_index]

    if det_type == "apple":
        #create transparent apple
        for apple in apple_list:
            result = np.zeros((100, 100, 3), np.uint8)
            apple_x = apple[0]
            apple_y = apple[1]
            bg = frames[0][apple_y:apple_y + apple_img.shape[0],
                           apple_x:apple_x + apple_img.shape[1]]

            alpha = apple_img[:, :, 3] / 255.0
            result[:, :, 0] = (1 - alpha) * bg[:, :, 0] + alpha * apple_img[:, :, 0]
            result[:, :, 1] = (1 - alpha) * bg[:, :, 1] + alpha * apple_img[:, :, 1]
            result[:, :, 2] = (1 - alpha) * bg[:, :, 2] + alpha * apple_img[:, :, 2]

            #added_image = cv2.addWeighted(bg, 0.4, apple_img, 0.3, 0)
            frames[0][apple_y:apple_y + apple_img.shape[0],
                      apple_x:apple_x + apple_img.shape[1]] = result

    cv2.imshow("detection", frames[0])
    del frames[0]
    key = cv2.waitKey(1)

    if key == ord('q'):
        end_det(device_index)
        sys.exit()

def convert_float_to_rgba(data, radix, platform, set_hwc=False):
    """Converts the NumPy float data into RGBA.

    Arguments:
        data: Input NumPy data
        radix: Radix of the input node
        platform: Integer platform (520 or 720)
        set_hwc: Flag to indicate if transpose is needed
    """
    if len(data.shape) == 3:    # add batch dimension
        data = np.reshape(data, (1, *data.shape))

    if set_hwc:                 # bchw -> bhwc
        data = np.transpose(data, (0, 2, 3, 1))
    _, height, width, channel = data.shape

    if platform == 520:
        width_aligned = 16 * math.ceil(width / 16.0)
    else:
        width_aligned = 4 * math.ceil(width / 4.0)

    aligned_data = np.zeros((1, height, width_aligned, 4))
    aligned_data[:, :height, :width, :channel] = data
    aligned_data *= 2 ** radix
    aligned_data = aligned_data.astype(np.int8)
    return aligned_data
