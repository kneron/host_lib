"""
This is the example for DME SSD FD single test.
"""
from common import constants, kdp_wrapper

def user_test_single_dme(dev_idx, loop):
    """Test single DME."""
    model_file = "../input_models/KL520/ssd_fd_lm/models_520.nef"
    model_id = constants.ModelType.KNERON_FD_MASK_MBSSD_200_200_3.value
    image_cfg = (constants.IMAGE_FORMAT_SUB128 | constants.NPU_FORMAT_RGB565)

    # Load model into Kneron device.
    dme_config = kdp_wrapper.init_dme_config(model_id, 8, image_cfg, ext_param=[0.5])
    ret = kdp_wrapper.dme_load_model(dev_idx, model_file, dme_config)
    if ret == -1:
        return -1

    image_source_h = 480
    image_source_w = 640
    image_size = image_source_w * image_source_h * 2
    frames = []
    app_id = constants.AppID.APP_FD_LM

    # Setup video capture device.
    capture = kdp_wrapper.setup_capture(0, image_source_w, image_source_h)
    if capture is None:
        return -1

    # Perform inference and draw result for each capture frame.
    while loop:
        det_res = kdp_wrapper.dme_inference(
            dev_idx, 0, capture, model_id, app_id, image_size, frames)
        if det_res is None:
            return -1

        kdp_wrapper.draw_capture_result(dev_idx, det_res, frames, "fd")
        loop -= 1

    return 0

def user_test(dev_idx, _user_id):
    # DME test
    ret = user_test_single_dme(dev_idx, 1000)
    kdp_wrapper.end_det(dev_idx)
    return ret
