"""
This is the example for cam isi ssd fd.
"""
import ctypes
import sys
import cv2
from common import constants_kl520
from examples_kl520.utils import kdp_wrapper


def handle_result(dev_idx, inf_res, r_size, frames):
    """Handle the detected results returned from the model.

    Arguments:
        dev_idx: Connected device ID. A host can connect several devices.
        inf_res: Inference result data.
        r_size: Inference data size.
        frames: List of frames captured by the video capture instance.
    """
    if r_size >= 4:
        header_result = ctypes.cast(
            ctypes.byref(inf_res), ctypes.POINTER(constants_kl520.ObjectDetectionRes)).contents
        # example of how to parse the detection results
        box_result = ctypes.cast(
            ctypes.byref(header_result.boxes),
            ctypes.POINTER(constants_kl520.BoundingBox * header_result.box_count)).contents
        # print("class_count", header_result.class_count)
        # print("box_count", header_result.box_count)
        for box in box_result:
            # print("box: %.3f, %.3f, %.3f, %.3f, %.6f, %d"
            #       % (b.x1, b.y1, b.x2, b.y2, b.score, b.class_num))
            x1 = int(box.x1)
            y1 = int(box.y1)
            x2 = int(box.x2)
            y2 = int(box.y2)
            class_num = int(box.class_num)
            if class_num == 1:
                frames[0] = cv2.rectangle(frames[0], (x1, y1), (x2, y2), (0, 0, 255), 3)
            if class_num == 2:
                frames[0] = cv2.rectangle(frames[0], (x1, y1), (x2, y2), (255, 0, 0), 3)

        cv2.imshow('detection', frames[0])
        del frames[0]
        key = cv2.waitKey(1)

        if key == ord('q'):
            kdp_wrapper.kdp_exit_isi(dev_idx)
            sys.exit()
    return 0

def user_test_cam_ssd_fd(dev_idx, _user_id, test_loop):
    """User test cam yolo."""
    image_source_h = 480
    image_source_w = 640
    app_id = constants_kl520.APP_FD_LM
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
    ret, img_id_tx, img_left, buffer_depth = kdp_wrapper.fill_buffer(
        dev_idx, capture, image_size, frames)
    if ret:
        return -1

    # Send the rest and get result in loop, with 2 images alternatively
    print("Companion image buffer depth = ", buffer_depth)
    ret = kdp_wrapper.pipeline_inference(
        dev_idx, app_id, test_loop, image_size,
        capture, img_id_tx, img_left, buffer_depth, frames, handle_result)

    kdp_wrapper.kdp_exit_isi(dev_idx)

    return ret

def user_test(dev_idx, user_id):
    """User test cam isi ssd fd"""
    user_test_cam_ssd_fd(dev_idx, user_id, 5000)
    return 0
