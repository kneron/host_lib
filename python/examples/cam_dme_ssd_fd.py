"""
This is the example for dme ssd fd single test.
"""
from python_wrapper import kdp_wrapper
from common import constants

import cv2
import sys


def overlap(x1_1,y1_1,x2_1,y2_1, x1_2,y1_2,x2_2,y2_2):
    #check if there is any overlap
    if (x1_1 > x2_2) or (x2_1 < x1_2):
        return 0
    elif (y1_1 > y2_2) or (y2_1 < y1_2):
        return 0

    #if there is overlap
    if (x1_1>x1_2):
        x1 = x1_1
    else:
        x1 = x1_2  

    if (y1_1>y1_2):
        y1 = y1_1
    else:
        y1 = y1_2 

    if (x2_1<x2_2):
        x2 = x2_1
    else:
        x2 = x2_2

    if (y2_1<y2_2):
        y2 = y2_1
    else:
        y2 = y2_2           

    w1 = x2_1 - x1_1
    h1 = y2_1 - y1_1
    w = x2 - x1
    h = y2 - y1
    print (w, h, w1, h1)
    o_l = float(w*h)/float(w1*h1)
    return o_l

def draw_result(dev_idx, det_res, captured_frames):
    x1_0 = 0
    y1_0 = 0
    x2_0 = 0
    y2_0 = 0
    score_0 = 0 
    #for multiple faces
    for res in det_res:
        #print(type(res))
        x1 = res[0]
        y1 = res[1]
        x2 = res[2]
        y2 = res[3]
        class_num = res[4]
        score = res[5]
        o_l = overlap(x1,y1,x2,y2,x1_0,y1_0,x2_0,y2_0)
        if (o_l<0.6):
            x1_0 = x1
            y1_0 = y1
            x2_0 = x2
            y2_0 = y2
            score_0 = score
            if (class_num==2):
                captured_frames[0] = cv2.rectangle(captured_frames[0], (x1, y1), (x2, y2), (0, 0, 255), 3)
                #print("score of mask fd: ", score)
            elif (class_num==1):
                captured_frames[0] = cv2.rectangle(captured_frames[0], (x1, y1), (x2, y2), (255, 0, 0), 3)   
                #print("score of fd: ", score)

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
    model_path = "../test_images/dme_ssd_fd"
    kdp_wrapper.kdp_dme_load_ssd_model(dev_idx, model_path)
    image_source_h = 480
    image_source_w = 640
    image_size = image_source_w * image_source_h * 2
    frames = []
    app_id = constants.APP_FD_LM

    # Setup video capture device.
    capture = kdp_wrapper.setup_capture(0, image_source_w, image_source_h)
    if capture is None:
        return -1

    while (loop):
        det_res = kdp_wrapper.kdp_dme_inference(dev_idx, app_id, capture, image_size, frames)
        draw_result(dev_idx, det_res, frames)
        loop -= 1
        # print("Total class {}: total detection {}".format(det_res[0], det_res[1]))
        # for i in range(det_res[1]):
        #     print("x1,y1,x2,y2:", det_res[4*i+2],det_res[4*i+3],det_res[4*i+4],det_res[4*i+5])

    kdp_wrapper.kdp_exit_dme(dev_idx)

def user_test_cam_dme_ssd_fd(dev_idx, user_id):
    # dme test
    user_test_single_dme(dev_idx, 1000)
    return 0
