"""
This is the example for the fcos example with pre and post in host side.
"""
import cv2
import numpy as np

from common import constants, kdp_wrapper
from common.pre_post_process.kneron_pre.kneron_preprocess import *
from common.pre_post_process.fcos.fcos_det_postprocess import *

INPUT_SHAPE_H = 416
INPUT_SHAPE_W = 416

def post_handler(dev_idx, raw_res, captured_frames_path):
    captured_frames = []
    captured_frames.append(cv2.imread(captured_frames_path))

    input_shape = (INPUT_SHAPE_H, INPUT_SHAPE_W)
    score_thres = 0.65
    max_objects = 100
    nms = True
    iou_thres = 0.5

    w_ori = 640
    h_ori = 480
    h = input_shape[0]
    w = input_shape[1]

    if w_ori > h or h_ori > w:
        scale = max(1.0*w_ori / w, 1.0*h_ori / h)
    else:
        scale = 1
    # print(scale)

    # reordering to the sequence which post function needs
    reordering = [2,5,8,0,3,6,1,4,7]
    raw_res = [raw_res[index] for index in reordering]

    transposed_results = []
    # do sigmoid for the cut-sigmoid nodes
    for i in range(len(raw_res)):
        new_result = raw_res[i].reshape(1, *raw_res[i].shape)
        if i > 2:
            new_result = kdp_wrapper.sigmoid(new_result)
        transposed_results.append(new_result)

    # As sigmoid nodes are cut, sigmoid operation is done for all output nodes in postprocess.
    det_res = postprocess_(transposed_results, max_objects, score_thres,
                        scale, input_shape, w_ori, h_ori, nms, iou_thres)
    #print(det_res)
    kdp_wrapper.draw_capture_result(dev_idx, det_res, captured_frames, "yolo", xywh=True)

    return 0

def pre_handler(frame):
    input_shape = (INPUT_SHAPE_H, INPUT_SHAPE_W)

    img_data, _im0 = preprocess(frame, INPUT_SHAPE_H, INPUT_SHAPE_W, True)
    return kdp_wrapper.convert_float_to_rgba(img_data, 8, 520, True)

def user_test_single_dme(dev_idx, loop):
    """Test single DME."""
    # As sigmoid is not supported by KL520 fw, the sigmoid nodes have been cut,
    # and NEF does not include sigmoid nodes.
    model_file = "../input_models/KL520/fcos_od_416/models_520.nef"
    model_id = constants.ModelType.CUSTOMER.value
    image_cfg = (constants.IMAGE_FORMAT_RAW_OUTPUT | constants.IMAGE_FORMAT_BYPASS_PRE)

    # Load model into Kneron device.
    dme_config = kdp_wrapper.init_dme_config(model_id, 3, image_cfg, image_row=416)
    ret = kdp_wrapper.dme_load_model(dev_idx, model_file, dme_config)
    if ret == -1:
        return -1

    # rgba input size
    image_size = INPUT_SHAPE_H * INPUT_SHAPE_W * 4
    app_id = 0 # if app_id is 0, output raw data for kdp_wrapper.kdp_dme_inference

    # to use images as inputs
    capture = None

    img_path = 'data/images/person_640x480.png'
    img_path2 = 'data/images/a_man_640x480.bmp'
    frames = [img_path, img_path2]

    # Send 1 image to the DME image buffers.
    ssid = kdp_wrapper.dme_fill_buffer(dev_idx, capture, image_size, frames, pre_handler)
    if ssid == -1:
        return -1

    return kdp_wrapper.dme_pipeline_inference(
        dev_idx, app_id, loop, image_size, capture, ssid, frames, post_handler, pre_handler)

def user_test(dev_idx, _user_id):
    # dme test
    ret = user_test_single_dme(dev_idx, 20)
    kdp_wrapper.end_det(dev_idx)
    return ret
