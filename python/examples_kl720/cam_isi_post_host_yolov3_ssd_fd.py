"""
This is the example for the yolov5 example.
"""
from common import constants, kdp_wrapper
from common.pre_post_process.yolo.yolo_postprocess import yolo_postprocess_
from common.pre_post_process.fdssd.ssd_postprocess import postprocess_

IMAGE_CHANNEL = 3
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
IMAGE_BPP = 2        # Byte Per Pixel = 2 for RGB565
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_BPP
IMAGE_FORMAT_YOLO = (constants.NPU_FORMAT_RGB565 | constants.IMAGE_FORMAT_RIGHT_SHIFT_ONE_BIT |
                     constants.IMAGE_FORMAT_RAW_OUTPUT)
IMAGE_FORMAT_FD = (constants.NPU_FORMAT_RGB565 | constants.IMAGE_FORMAT_SUB128 |
                   constants.IMAGE_FORMAT_RAW_OUTPUT)

# APP_CENTER_APP always works for single model.
APP_ID = constants.AppID.APP_CENTER_APP.value
MODEL_ID_YOLO = constants.ModelType.CUSTOMER.value
MODEL_ID_FD = constants.ModelType.KNERON_FD_MBSSD_200_200_3.value

def do_inference(device_index, capture, buf_len, img_id, frames, model_id, image_format):
    """Performs inference.

    Arguments:
        device_index: Integer connected device ID
        capture: Active cv2 video capture instance
        buf_len: Integer image buffer size
        img_id: Integer ID of the test image
        frames: List of frames for the video capture to add to
        model_id: Integer model ID to perform inference
        image_format: Integer input image format
    """
    image_info = constants.KDPISIImageHeader(
        buf_len, img_id, IMAGE_WIDTH, IMAGE_HEIGHT, image_format, model_id)
    img_left = kdp_wrapper.isi_inference_ext(
        device_index, image_info, capture, buf_len, img_id, frames)

    if img_left == -1:
        return -1
    return 0

def user_test_single_yolo(device_index, loop):
    model_file = "../input_models/KL720/yolo_v3_416_ssd_fd/models_720.nef"

    isi_data = kdp_wrapper.init_isi_data(
        IMAGE_CHANNEL, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_FORMAT_YOLO, MODEL_ID_YOLO)

    # Load NEF model into board.
    ret = kdp_wrapper.isi_load_nef(device_index, model_file, APP_ID, isi_data=isi_data)

    if ret:
        print("Load example 720 yolov3 model failed...")
        return ret

    img_id_tx = 1
    img_id_rx = img_id_tx
    buf_len = 640 * 480 * 2
    frames = []

    # the parameters for postprocess
    anchor_path_yolo = './common/pre_post_process/yolo/models/anchors_v3.txt'
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
    capture = kdp_wrapper.setup_capture(0, IMAGE_WIDTH, IMAGE_HEIGHT)
    if capture is None:
        return -1

    img_id = 1
    while loop:
        if loop // 20 % 2:
            ret = do_inference(device_index, capture, buf_len, img_id_tx,
                               frames, MODEL_ID_YOLO, IMAGE_FORMAT_YOLO)
            if ret:
                return ret
            img_id_tx += 1

            # Get results.
            result_data, _result_size = kdp_wrapper.isi_retrieve_res(device_index, img_id_rx)

            if result_data is None:
                return -1

            # output will be in (h,w,c) format for postprocess
            np_results = kdp_wrapper.convert_data_to_numpy(
                result_data, add_batch=False, channel_last=True)

            img_id_rx += 1

            dets = yolo_postprocess_(
                np_results, anchor_path_yolo, class_path, IMAGE_HEIGHT, IMAGE_WIDTH,
                model_input_shape_yolo, score_thres_yolo, nms_thres_yolo, keep_aspect_ratio)

            kdp_wrapper.draw_capture_result(device_index, dets, frames, "yolo", xywh=False)
        else:
            ret = do_inference(device_index, capture, buf_len, img_id_tx,
                               frames, MODEL_ID_FD, IMAGE_FORMAT_FD)
            if ret:
                return ret
            img_id_tx += 1

            # Get results.
            result_data, _result_size = kdp_wrapper.isi_retrieve_res(device_index, img_id_rx)

            if result_data is None:
                return -1

            # output will be in (c, h, w) format
            np_results = kdp_wrapper.convert_data_to_numpy(
                result_data, add_batch=False, channel_last=True)

            img_id_rx += 1

            np_results.sort(key=lambda x: x.size)
            dets = postprocess_(np_results, anchor_path, model_input_shape, IMAGE_WIDTH,
                                IMAGE_HEIGHT, score_thres, only_max, nms_thres)

            kdp_wrapper.draw_capture_result(
                device_index, dets, frames, "fd_mask_no_overlap", xywh=True)
        img_id += 1
        loop -= 1
    return 0

def user_test(device_index, _user_id):
    ret = user_test_single_yolo(device_index, 100)
    kdp_wrapper.end_det(device_index)
    return ret
