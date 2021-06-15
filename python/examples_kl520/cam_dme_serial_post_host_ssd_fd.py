"""
This is the example for DME SSD FD single test.
"""
from common import constants, kdp_wrapper
from common.pre_post_process.fdssd.ssd_postprocess import postprocess_

def user_test_single_dme(dev_idx, loop):
    """Test single DME."""
    model_file = "../input_models/KL520/ssd_fd_lm/models_520.nef"
    model_id = constants.ModelType.KNERON_FD_MASK_MBSSD_200_200_3.value
    image_cfg = (constants.IMAGE_FORMAT_SUB128 | constants.NPU_FORMAT_RGB565 |
                    constants.IMAGE_FORMAT_RAW_OUTPUT)

    # Load model into Kneron device.
    dme_config = kdp_wrapper.init_dme_config(model_id, 8, image_cfg, ext_param=[0.5])
    ret = kdp_wrapper.dme_load_model(dev_idx, model_file, dme_config)
    if ret == -1:
        return -1

    image_source_h = 480
    image_source_w = 640
    image_size = image_source_w * image_source_h * 2
    frames = []
    app_id = 0 # if app_id is 0, output raw data for kdp_wrapper.kdp_dme_inference

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

    # Perform inference and draw result for each capture frame.
    while loop:
        raw_res = kdp_wrapper.dme_inference(
            dev_idx, 0, capture, model_id, app_id, image_size, frames)
        if raw_res is None:
            return -1

        det_res = postprocess_(raw_res, anchor_path, model_input_shape, image_source_w,
                               image_source_h, score_thres, only_max, nms_thres)
        #print(det_res)
        kdp_wrapper.draw_capture_result(dev_idx, det_res, frames, "fd_mask_no_overlap", xywh=True)
        loop -= 1

    return 0

def user_test(dev_idx, _user_id):
    # DME test
    ret = user_test_single_dme(dev_idx, 1000)
    kdp_wrapper.end_det(dev_idx)
    return ret
