"""
This is the 720 classification example with preprocessing and postprocessing done on host side.
"""
import pathlib

import numpy as np

from common import constants, kdp_wrapper
from common.pre_post_process.classification.classification_postprocess import postprocess
from common.pre_post_process.kneron_pre.kneron_preprocess import preprocess

# Example model/image paths
ROOT_FOLDER = pathlib.Path(__file__).resolve().parent.parent.parent

IMAGE_FOLDER = ROOT_FOLDER / "python/data/images"
IMAGE_CAT = str(IMAGE_FOLDER / "cat.jpg")
IMAGE_FOX = str(IMAGE_FOLDER / "fox.jpg")

MODEL_FOLDER = ROOT_FOLDER / "input_models/KL720/classification"
MODEL_FILE = str(MODEL_FOLDER / "models_720.nef")

# RGBA input image configurations - same as model input dimensions
IMAGE_CHANNEL = 3
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_BPP = 4
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_BPP
# RAW_OUTPUT for host postprocess, BYPASS_PRE for host preprocess
IMAGE_FORMAT = constants.IMAGE_FORMAT_RAW_OUTPUT | constants.IMAGE_FORMAT_BYPASS_PRE

# App ID of APP_CENTER_APP will always works for a single model
APP_ID = constants.AppID.APP_CENTER_APP.value

# Model ID is the same one generated with batch compile (32768 in this case)
MODEL_ID = constants.ModelType.CUSTOMER.value

def top_indices(preds, num):
    """Print indices and scores of the top num scores."""
    sort_preds = np.sort(preds, 1)
    sort_preds = np.flip(sort_preds)
    sort_index = np.argsort(preds, 1)
    sort_index = np.flip(sort_index)

    print(f"Top {num} results:")
    for i in range(num):
        print(sort_index[0][i], sort_preds[0][i])

    return 0

def post_handler(raw_res, img_type):
    """Function handler to perform host postprocessing."""
    if img_type == "cat":
        print("\nCat classification:")
    else:
        print("\nFox classification:")
    scores, _labels = postprocess(raw_res[0])
    top_indices(scores[0], 3)

def pre_handler(input_image):
    """Function handler to perform host preprocessing."""
    img_data, _im0 = preprocess(input_image, IMAGE_HEIGHT, IMAGE_WIDTH, False)
    return kdp_wrapper.convert_float_to_rgba(img_data, 8, 720, True)

def user_test_single_classification(device_index, loop):
    """Test single ISI."""
    # Load NEF model into board.
    isi_data = kdp_wrapper.init_isi_data(
        IMAGE_CHANNEL, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_FORMAT, MODEL_ID)
    ret = kdp_wrapper.isi_load_nef(device_index, MODEL_FILE, APP_ID, isi_data=isi_data)
    if ret:
        return ret

    img_id = 1
    while loop:
        if loop % 2:
            cur_image = IMAGE_CAT
            img_type = "cat"
        else:
            cur_image = IMAGE_FOX
            img_type = "fox"

        # Preprocess.
        img_data = pre_handler(cur_image)

        # Inference the image.
        img_buf = kdp_wrapper.convert_numpy_to_char_p(img_data, size=IMAGE_SIZE)
        img_left = kdp_wrapper.isi_inference(device_index, img_buf, IMAGE_SIZE, img_id)
        if img_left == -1:
            return -1

        # Get results.
        result_data, _result_size = kdp_wrapper.isi_retrieve_res(device_index, img_id)
        if result_data is None:
            return -1

        # Output will be in (1, h, w, c) format
        raw_res = kdp_wrapper.convert_data_to_numpy(
            result_data, add_batch=True, channel_last=True)

        # Postprocess.
        post_handler(raw_res[0], img_type)

        img_id += 1
        loop -= 1
    return 0

def user_test(device_index, _user_id):
    """ISI test."""
    ret = user_test_single_classification(device_index, 20)
    kdp_wrapper.end_det(device_index)
    return ret
