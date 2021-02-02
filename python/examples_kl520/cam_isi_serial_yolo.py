"""
This is the example for cam isi serial yolo.
"""
from examples_kl520.utils import kdp_wrapper
import cv2
import ctypes
import time
import sys
from common import constants_kl520
ISI_YOLO_ID = constants_kl520.APP_TINY_YOLO3

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
        box_result = ctypes.cast(
            ctypes.byref(header_result.boxes),
            ctypes.POINTER(constants_kl520.BoundingBox * header_result.box_count)).contents
        for box in box_result:
            x1 = int(box.x1)
            y1 = int(box.y1)
            x2 = int(box.x2)
            y2 = int(box.y2)
            frames[0] = cv2.rectangle(frames[0], (x1, y1), (x2, y2), (0, 0, 255), 3)

        cv2.imshow('detection', frames[0])
        del frames[0]
        key = cv2.waitKey(1)

        if key == ord('q'):
            kdp_wrapper.kdp_exit_isi(dev_idx)
            sys.exit()
    return 0

def user_test_cam_sync_yolo(dev_idx, _user_id, test_loop):
    """User test cam yolo."""
    image_source_w = 640
    image_source_h = 480
    image_size = image_source_w * image_source_h * 2
    frames = []

    # Setup video capture device.
    capture = kdp_wrapper.setup_capture(0, image_source_w, image_source_h)
    if capture is None:
        return -1

    # Start ISI mode.
    if kdp_wrapper.start_isi(dev_idx, ISI_YOLO_ID, image_source_w, image_source_h):
        return -1

    # Fill up the image buffers.
    img_id_tx = 0
    start_time = time.time()
    while (img_id_tx!=test_loop):
        kdp_wrapper.sync_inference(dev_idx, ISI_YOLO_ID, image_size, capture,
                                   img_id_tx, frames, handle_result)
        img_id_tx+=1
    end_time = time.time()
    diff = end_time - start_time 
    estimate_runtime = float(diff/test_loop)
    fps = float(1/estimate_runtime)    
    print("Sync inference average estimate runtime is ", estimate_runtime)
    print("Average FPS is ", fps)

    kdp_wrapper.kdp_exit_isi(dev_idx)

    return 0



def user_test(dev_idx, user_id):
    """User test cam isi yolo"""
    user_test_cam_sync_yolo(dev_idx, user_id, 1000)
    return 0