"""
This is the 720 public Yolov3 example with postprocessing done on host side.
"""
import pathlib

from common import constants, kdp_wrapper
from common.pre_post_process.yolo.yolo_postprocess import yolo_postprocess_

# Example model/image paths
ROOT_FOLDER = pathlib.Path(__file__).resolve().parent.parent.parent

ANCHOR_FOLDER = ROOT_FOLDER / "python/common/pre_post_process/yolo/models"
ANCHOR_PATH = str(ANCHOR_FOLDER / "anchors_v3.txt")

MODEL_FOLDER = ROOT_FOLDER / "input_models/KL720/public_yolov3_416"
MODEL_PATH = str(MODEL_FOLDER / "models_720.nef")

CLASS_FOLDER = ROOT_FOLDER / "python/common/class_lists"
CLASS_PATH = str(CLASS_FOLDER / "coco_name_lists")

IMAGE_FOLDER = ROOT_FOLDER / "input_images"
IMAGE_PATH = str(IMAGE_FOLDER / "a_man2_640x480_rgb565.bin")

# RGB565 input image configurations
IMAGE_CHANNEL = 3
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
IMAGE_BPP = 2
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_BPP
# RGB565 input binary, radix = 7 for YOLO model, RAW_OUTPUT for host postproces
IMAGE_FORMAT = (constants.NPU_FORMAT_RGB565 | constants.IMAGE_FORMAT_RIGHT_SHIFT_ONE_BIT |
                constants.IMAGE_FORMAT_RAW_OUTPUT)

# Postprocess parameters
MODEL_HEIGHT = 416
MODEL_WIDTH = 416
SCORE_THRESHOLD = 0.3
NMS_THRESHOLD = 0.45

# App ID of APP_CENTER_APP will always works for a single model
APP_ID = constants.AppID.APP_CENTER_APP.value

# Model ID is the same one generated with batch compile (32768 in this case)
MODEL_ID = constants.ModelType.CUSTOMER.value

def user_test_single_yolo(device_index):
    """Test single ISI."""
    # Load NEF model into board.
    isi_data = kdp_wrapper.init_isi_data(
        IMAGE_CHANNEL, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_FORMAT, MODEL_ID)
    ret = kdp_wrapper.isi_load_nef(device_index, MODEL_PATH, APP_ID, isi_data=isi_data)
    if ret:
        return ret

    # Read image file.
    img_buf, length = kdp_wrapper.read_file_to_buf_with_size(IMAGE_PATH, IMAGE_SIZE)
    if img_buf is None:
        return -1

    # Inference the image.
    img_id = 1
    img_left = kdp_wrapper.isi_inference(device_index, img_buf, length, img_id)
    if img_left == -1:
        return -1

    # Get results.
    result_data, _result_size = kdp_wrapper.isi_retrieve_res(device_index, img_id)
    if result_data is None:
        return -1

    # Output will be in (1, h, w, c) format
    np_results = kdp_wrapper.convert_data_to_numpy(result_data, add_batch=True, channel_last=True)

    dets = yolo_postprocess_(np_results, ANCHOR_PATH, CLASS_PATH, IMAGE_HEIGHT, IMAGE_WIDTH,
                             (MODEL_WIDTH, MODEL_HEIGHT), SCORE_THRESHOLD, NMS_THRESHOLD, True)
    print(dets)
    kdp_wrapper.display(IMAGE_PATH, dets, CLASS_PATH, det_type="xyxy",
                        image_format="rgb565", size=(IMAGE_WIDTH, IMAGE_HEIGHT))

    return 0

def user_test(device_index, _user_id):
    """ISI test."""
    ret = user_test_single_yolo(device_index)
    kdp_wrapper.end_det(device_index)
    return ret
