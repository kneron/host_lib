"""
This is the 520 classification example with preprocessing and postostprocessing done on host side.
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

MODEL_FOLDER = ROOT_FOLDER / "input_models/KL520/classification"
MODEL_FILE = str(MODEL_FOLDER / "models_520.nef")

# Model ID is the same one generated with batch compile (32768 in this case)
MODEL_ID = constants.ModelType.CUSTOMER.value

# App ID of 0 returns output raw data when calling kdp_wrapper.kdp_dme_inference()
APP_ID = 0

# RGBA input image configurations - same as model input dimensions
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_BPP = 4
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_BPP
# RAW_OUTPUT for host postprocess, BYPASS_PRE for host preprocess
IMAGE_FORMAT = constants.IMAGE_FORMAT_RAW_OUTPUT | constants.IMAGE_FORMAT_BYPASS_PRE

# Used to keep track of current image
CUR_IMAGE = 0

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

def post_handler(_dev_idx, raw_res, _captured_frames_path):
    """Function handler to perform host postprocessing."""
    global CUR_IMAGE
    if CUR_IMAGE % 2:
        print("\nFox classification")
    else:
        print("\nCat classification")

    scores, _labels = postprocess(raw_res[0])
    top_indices(scores[0], 3)

    CUR_IMAGE += 1

def pre_handler(frame):
    """Function handler to perform host preprocessing."""
    img_data, _im0 = preprocess(frame, IMAGE_HEIGHT, IMAGE_WIDTH, False)
    return kdp_wrapper.convert_float_to_rgba(img_data, 8, 520, True)

def user_test_single_dme(dev_idx, loop):
    """Test single DME."""
    # Load model into Kneron device.
    dme_config = kdp_wrapper.init_dme_config(
        MODEL_ID, 1, IMAGE_FORMAT, image_col=IMAGE_WIDTH, image_row=IMAGE_HEIGHT)
    ret = kdp_wrapper.dme_load_model(dev_idx, MODEL_FILE, dme_config)
    if ret == -1:
        return -1

    # If capture is None, inference will use frames as input images.
    capture = None
    frames = [IMAGE_CAT, IMAGE_FOX]

    # Send 1 image to the DME image buffers.
    ssid = kdp_wrapper.dme_fill_buffer(dev_idx, capture, IMAGE_SIZE, frames, pre_handler)
    if ret == -1:
        return -1

    # Perform inference on the image set 'loop' times.
    return kdp_wrapper.dme_pipeline_inference(
        dev_idx, APP_ID, loop, IMAGE_SIZE, capture, ssid, frames, post_handler, pre_handler)

def user_test(dev_idx, _user_id):
    """DME test."""
    ret = user_test_single_dme(dev_idx, 20)
    kdp_wrapper.end_det(dev_idx)
    return ret
