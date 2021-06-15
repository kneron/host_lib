"""
This is the example for cam ISI SSD FD.
"""
from common import constants, kdp_wrapper

def handle_result(dev_idx, inf_res, r_size, frames):
    """Handle the detected results returned from the model.

    Arguments:
        dev_idx: Integer connected device ID
        inf_res: Inference result data, should be result of isi_get_result()
        r_size: Integer inference data size
        frames: List of frames captured by the video capture instance
    """
    if r_size >= 4:
        header_result = kdp_wrapper.cast_and_get(inf_res, constants.ObjectDetectionRes)
        box_result = kdp_wrapper.cast_and_get(
            header_result.boxes, constants.BoundingBox * header_result.box_count)

        boxes = []
        for box in box_result:
            boxes.append([box.x1, box.y1, box.x2, box.y2, box.score, box.class_num])
        kdp_wrapper.draw_capture_result(dev_idx, boxes, frames, "fd_mask_no_overlap")

    return 0

def user_test_cam_ssd_fd(dev_idx, _user_id, test_loop):
    """User test cam yolo."""
    image_source_h = 480
    image_source_w = 640
    app_id = constants.AppID.APP_FD_LM
    image_size = image_source_w * image_source_h * 2
    frames = []

    # Setup video capture device.
    capture = kdp_wrapper.setup_capture(0, image_source_w, image_source_h)
    if capture is None:
        return -1

    # Start ISI mode.
    if kdp_wrapper.start_isi(dev_idx, app_id, image_source_w, image_source_h):
        return -1

    # Fill up the image buffers.
    ret, img_id_tx, img_left, buffer_depth = kdp_wrapper.isi_fill_buffer(
        dev_idx, capture, image_size, frames)
    if ret:
        return -1

    # Send the rest and get result in loop, with 2 images alternatively
    print("Companion image buffer depth = ", buffer_depth)
    ret = kdp_wrapper.isi_pipeline_inference(
        dev_idx, app_id, test_loop, image_size,
        capture, img_id_tx, img_left, buffer_depth, frames, handle_result)

    return ret

def user_test(dev_idx, user_id):
    """User test cam isi ssd fd"""
    ret = user_test_cam_ssd_fd(dev_idx, user_id, 5000)
    kdp_wrapper.end_det(dev_idx)
    return ret
