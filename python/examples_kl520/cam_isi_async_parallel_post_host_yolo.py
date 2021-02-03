"""
This is the example for cam isi yolo.
"""
import ctypes
import sys
import cv2
import time
from common import constants_kl520
from examples_kl520.utils import kdp_wrapper
from examples_kl520.yolo.yolo_postprocess import yolo_postprocess_


def handle_result(dev_idx, raw_res, captured_frames):
    """Handle the raw results returned from the model.

    Arguments:
        dev_idx: Connected device ID. A host can connect several devices.
        raw_res: Raw numpy data.
        captured_frames: List of frames captured by the video capture instance.
    """
    # the parameters for postprocess
    anchor_path       = './examples_kl520/yolo/models/anchors.txt'
    class_path        = './common/coco_name_lists'
    model_input_shape = (224, 224)
    score_thres       = 0.2
    nms_thres         = 0.45
    keep_aspect_ratio = True
    image_source_h    = 480
    image_source_w    = 640

    det_res = yolo_postprocess_(raw_res, anchor_path, class_path, image_source_h, image_source_w, model_input_shape,
                             score_thres, nms_thres, keep_aspect_ratio)
    #for multiple detection
    for res in det_res:
        x1 = int(res[0])
        y1 = int(res[1])
        x2 = int(res[2])
        y2 = int(res[3])
        class_num = res[5]
        score = res[4]
        # print(x1,x2,class_num,score)

        if (class_num==0):
            captured_frames[0] = cv2.rectangle(captured_frames[0], (x1, y1), (x2, y2), (0, 0, 255), 3)
            # print("score of person: ", score)
        else:
            captured_frames[0] = cv2.rectangle(captured_frames[0], (x1, y1), (x2, y2), (255, 0, 0), 3)
            # print("score of others: ", score)

    cv2.imshow('detection', captured_frames[0])
    del captured_frames[0]
    key = cv2.waitKey(1)

    if key == ord('q'):
        kdp_wrapper.kdp_exit_isi(dev_idx)
        sys.exit()

    return 

def user_test_cam_yolo(dev_idx, _user_id, test_loop):
    """User test cam yolo."""
    is_raw_output = True
    image_source_h = 480
    image_source_w = 640
    app_id = constants_kl520.APP_TINY_YOLO3
    image_size = image_source_w * image_source_h * 2
    frames = []

    # Setup video capture device.
    capture = kdp_wrapper.setup_capture(0, image_source_w, image_source_h)
    if capture is None:
        return -1

    # Start ISI mode.
    if kdp_wrapper.start_isi_parallel_ext(dev_idx, app_id, image_source_w, image_source_h, is_raw_output):
        return -1

    start_time = time.time()
    # Fill up the image buffers.
    ret, img_id_tx, img_left, buffer_depth = kdp_wrapper.fill_buffer(
        dev_idx, capture, image_size, frames)
    if ret:
        return -1

    # Send the rest and get result in loop, with 2 images alternatively
    print("Companion image buffer depth = ", buffer_depth)
    kdp_wrapper.pipeline_inference(
        dev_idx, app_id, test_loop - buffer_depth, image_size,
        capture, img_id_tx, img_left, buffer_depth, frames, handle_result, is_raw_output)

    end_time = time.time()
    diff = end_time - start_time 
    estimate_runtime = float(diff/test_loop)
    fps = float(1/estimate_runtime)    
    print("Parallel inference average estimate runtime is ", estimate_runtime)
    print("Average FPS is ", fps)

    kdp_wrapper.kdp_exit_isi(dev_idx)

    return 0    

def user_test(dev_idx, user_id):
    """User test cam isi yolo."""
    #for i in range(10):
    user_test_cam_yolo(dev_idx, user_id, 1000)
    return    
