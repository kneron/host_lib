"""
This is the example for the fcos example with pre and post in host side.
"""
import numpy as np
from common import constants, kdp_wrapper
from common.pre_post_process.kneron_pre.kneron_preprocess import *
from common.pre_post_process.fcos.fcos_det_postprocess import *

IMAGE_CHANNEL = 3
IMAGE_WIDTH = 416
IMAGE_HEIGHT = 416
IMAGE_BPP = 4        # Byte Per Pixel = 4 for RGBA
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_BPP
IMAGE_FORMAT = (constants.IMAGE_FORMAT_RAW_OUTPUT | constants.IMAGE_FORMAT_BYPASS_PRE)

# APP_CENTER_APP always works for single model.
APP_ID = constants.AppID.APP_CENTER_APP.value
MODEL_ID = constants.ModelType.CUSTOMER.value

def user_test_single_yolo(device_index, loop):
    model_file = "../input_models/KL720/fcos_od_416/models_720.nef"

    isi_data = kdp_wrapper.init_isi_data(
        IMAGE_CHANNEL, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_FORMAT, MODEL_ID)

    # Load NEF model into board.
    ret = kdp_wrapper.isi_load_nef(device_index, model_file, APP_ID, isi_data=isi_data)

    if ret:
        print("Load example 720 yolov3 model failed...")
        return ret

    image_source_h = 480
    image_source_w = 640

    frames = []

    # Setup video capture device.
    capture = kdp_wrapper.setup_capture(0, image_source_w, image_source_h)
    if capture is None:
        return -1

    keep_aspect_ratio = True
    img_id = 1
    while loop:
        kdp_wrapper.isi_capture_frame(capture, frames)
        # preprocess
        input_shape = (416, 416)
        img_data, im0 = preprocess(frames[0], 416, 416, keep_aspect_ratio)

        img_data = kdp_wrapper.convert_float_to_rgba(img_data, 8, 720, True)

        # Inference the image.
        img_buf = kdp_wrapper.convert_numpy_to_char_p(img_data, size=IMAGE_SIZE)
        img_left = kdp_wrapper.isi_inference(device_index, img_buf, IMAGE_SIZE, img_id)

        if img_left == -1:
            return -1

        # Get results.
        result_data, _result_size = kdp_wrapper.isi_retrieve_res(device_index, img_id)

        if result_data is None:
            return -1

        # output will be in (1, h, w, c) format
        raw_res = kdp_wrapper.convert_data_to_numpy(result_data, add_batch=True, channel_last=True)

        score_thres = 0.6
        max_objects = 100
        nms = True
        iou_thres = 0.5
        w_ori = image_source_w
        h_ori = image_source_h

        h = input_shape[0]
        w = input_shape[1]

        if w_ori > h or h_ori > w:
            scale = max(1.0*w_ori / w, 1.0*h_ori / h)
        else:
            scale = 1
        # reordering to the sequence which post function need
        reordering = [1,4,7,2,5,8,0,3,6]
        raw_res = [raw_res[index] for index in reordering]

        dets = postprocess_(raw_res, max_objects, score_thres,
                            scale, input_shape, w_ori, h_ori, nms, iou_thres)

        # print("dets: ", dets)
        kdp_wrapper.draw_capture_result(device_index, dets, frames, "yolo", xywh=True)
        img_id += 1
        loop -= 1
    return 0

def user_test(device_index, user_id):
    ret = user_test_single_yolo(device_index, 20)
    kdp_wrapper.end_det(device_index)
    return ret
