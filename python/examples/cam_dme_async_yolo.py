"""
This is the example for dme ssd fd single test.
"""
from python_wrapper import kdp_wrapper
from common import constants
from examples.yolo.yolo_postprocess import yolo_postprocess_
import cv2
import sys

def handle_result(dev_idx, raw_res, captured_frames):
    # the parameters for postprocess
    anchor_path       = './examples/yolo/models/anchors.txt'
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
        kdp_wrapper.kdp_exit_dme(dev_idx)
        sys.exit()

    return 

def user_test_single_dme(dev_idx, loop):
    """Test single dme."""
    # load model into Kneron device
    model_path = "../test_images/dme_yolo_224"
    kdp_wrapper.kdp_dme_load_yolo_model(dev_idx, model_path)

    image_source_h = 480
    image_source_w = 640
    image_size = image_source_w * image_source_h * 2
    frames = []
    app_id = constants.APP_TINY_YOLO3

    # Setup video capture device.
    capture = kdp_wrapper.setup_capture(0, image_source_w, image_source_h)
    if capture is None:
        return -1

    # Send 1 image to the DME image buffers.
    ret, ssid = kdp_wrapper.dme_fill_buffer(dev_idx, capture, image_size, frames)
    if ret:
        return -1

    kdp_wrapper.dme_pipeline_inference(dev_idx, app_id, loop, image_size, capture, ssid, frames, handle_result)

    kdp_wrapper.kdp_exit_dme(dev_idx)

def user_test_cam_dme_async_yolo(dev_idx, user_id):
    # dme test
    user_test_single_dme(dev_idx, 1000)
    return 0
