"""
This is the serial example for switching between tiny YOLOv3 and SSD FD.
"""
from common import constants, kdp_wrapper
from common.pre_post_process.yolo.yolo_postprocess import yolo_postprocess_
from common.pre_post_process.fdssd.ssd_postprocess import postprocess_

def user_test_single_dme(dev_idx, loop):
    """Test single DME."""
    model_file = "../input_models/KL520/tiny_yolo_v3_416_ssd_fd/models_520.nef"
    model_id = constants.ModelType.CUSTOMER.value
    image_cfg = (constants.IMAGE_FORMAT_SUB128 | constants.NPU_FORMAT_RGB565 |
                    constants.IMAGE_FORMAT_RAW_OUTPUT)

    # Load model into Kneron device.
    dme_config = kdp_wrapper.init_dme_config(model_id, 1, image_cfg, ext_param=[0.5])
    ret = kdp_wrapper.dme_load_model(dev_idx, model_file, dme_config)
    if ret:
        return -1

    image_source_h = 480
    image_source_w = 640
    image_size = image_source_w * image_source_h * 2
    frames = []
    app_id = 0

    # the parameters for postprocess
    anchor_path_yolo = './common/pre_post_process/yolo/models/anchors_tiny_v3.txt'
    class_path = './common/class_lists/coco_name_lists'
    model_input_shape_yolo = (416, 416)
    score_thres_yolo = 0.1
    nms_thres_yolo = 0.45
    keep_aspect_ratio = True

    # the parameters for postprocess
    anchor_path = './common/pre_post_process/fdssd/models/anchor_face.npy'
    model_input_shape = (200, 200)
    score_thres = 0.5
    nms_thres = 0.35
    only_max = False

    # Setup video capture device.
    capture = kdp_wrapper.setup_capture(0, image_source_w, image_source_h)
    if capture is None:
        return -1

    # Initialize DME configurations for both models
    yolo_id = constants.ModelType.CUSTOMER.value
    fd_id = constants.ModelType.KNERON_FD_MBSSD_200_200_3.value
    yolo_format = (constants.NPU_FORMAT_RGB565 | constants.IMAGE_FORMAT_RIGHT_SHIFT_ONE_BIT |
                   constants.IMAGE_FORMAT_RAW_OUTPUT)
    fd_format = (constants.IMAGE_FORMAT_SUB128 | constants.NPU_FORMAT_RGB565 |
                 constants.IMAGE_FORMAT_RAW_OUTPUT)

    yolo_config = kdp_wrapper.init_dme_config(
        yolo_id, 2, yolo_format, image_source_w, image_source_h, 3, [0.5])
    fd_config = kdp_wrapper.init_dme_config(
        fd_id, 2, fd_format, image_source_w, image_source_h, 3, [0.5])

    # total_loop = loop
    while loop:
        if loop // 10 % 2:
            kdp_wrapper.dme_configure(dev_idx, yolo_config, False)
            raw_res = kdp_wrapper.dme_inference(
                dev_idx, 0, capture, yolo_id, app_id, image_size, frames)
            if raw_res is None:
                return -1

            dets = yolo_postprocess_(
                raw_res, anchor_path_yolo, class_path, image_source_h, image_source_w,
                model_input_shape_yolo, score_thres_yolo, nms_thres_yolo, keep_aspect_ratio)

            # print(dets)
            kdp_wrapper.draw_capture_result(dev_idx, dets, frames, "yolo")
        else:
            kdp_wrapper.dme_configure(dev_idx, fd_config, False)
            raw_res = kdp_wrapper.dme_inference(
                dev_idx, 0, capture, fd_id, app_id, image_size, frames)
            if raw_res is None:
                return -1

            dets = postprocess_(raw_res, anchor_path, model_input_shape, image_source_w,
                                image_source_h, score_thres, only_max, nms_thres)

            # print(dets)
            kdp_wrapper.draw_capture_result(dev_idx, dets, frames, "fd_no_overlap", xywh=True)

        loop -= 1

    return 0

def user_test(dev_idx, _user_id):
    # DME test
    ret = user_test_single_dme(dev_idx, 100)
    kdp_wrapper.end_det(dev_idx)
    return ret
