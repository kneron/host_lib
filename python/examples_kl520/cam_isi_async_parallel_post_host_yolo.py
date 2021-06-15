"""
This is the example for cam ISI YOLO.
"""
import time

from common import constants, kdp_wrapper
from common.pre_post_process.yolo.yolo_postprocess import yolo_postprocess_

def handle_result(dev_idx, raw_res, captured_frames):
    """Handle the raw results returned from the model.

    Arguments:
        dev_idx: Integer connected device ID
        raw_res: Raw numpy data, should be result of isi_get_result()
        captured_frames: List of frames captured by the video capture instance
    """
    # the parameters for postprocess
    anchor_path = './common/pre_post_process/yolo/models/anchors_tiny_v3.txt'
    class_path = './common/class_lists/coco_name_lists'
    model_input_shape = (224, 224)
    score_thres = 0.2
    nms_thres = 0.45
    keep_aspect_ratio = True
    image_source_h = 480
    image_source_w = 640

    det_res = yolo_postprocess_(raw_res, anchor_path, class_path, image_source_h, image_source_w,
                                model_input_shape, score_thres, nms_thres, keep_aspect_ratio)
    kdp_wrapper.draw_capture_result(dev_idx, det_res, captured_frames, "yolo")

    return 0

def user_test_cam_yolo(dev_idx, _user_id, test_loop):
    """User test cam yolo."""
    is_raw_output = True
    image_source_h = 480
    image_source_w = 640
    app_id = constants.AppID.APP_TINY_YOLO3
    image_size = image_source_w * image_source_h * 2
    frames = []

    # Setup video capture device.
    capture = kdp_wrapper.setup_capture(0, image_source_w, image_source_h)
    if capture is None:
        return -1

    # Start ISI mode.
    if kdp_wrapper.start_isi_parallel_ext(
            dev_idx, app_id, image_source_w, image_source_h, is_raw_output):
        return -1

    start_time = time.time()
    # Fill up the image buffers.
    ret, img_id_tx, img_left, buffer_depth = kdp_wrapper.isi_fill_buffer(
        dev_idx, capture, image_size, frames)
    if ret:
        return -1

    # Send the rest and get result in loop, with 2 images alternatively
    print("Companion image buffer depth = ", buffer_depth)
    ret = kdp_wrapper.isi_pipeline_inference(
        dev_idx, app_id, test_loop - buffer_depth, image_size,
        capture, img_id_tx, img_left, buffer_depth, frames, handle_result, is_raw_output)

    end_time = time.time()
    diff = end_time - start_time
    estimate_runtime = float(diff / test_loop)
    fps = float(1 / estimate_runtime)
    print("Parallel inference average estimate runtime is ", estimate_runtime)
    print("Average FPS is ", fps)

    return ret

def user_test(dev_idx, user_id):
    """User test cam ISI YOLO."""
    ret = user_test_cam_yolo(dev_idx, user_id, 1000)
    kdp_wrapper.end_det(dev_idx)
    return ret
