"""
This is the example for DME Keras single test.
"""
from common import constants, kdp_wrapper
import numpy as np

def top_indexes(preds, n):
    sort_preds = np.sort(preds,1)
    sort_preds = np.flip(sort_preds)
    sort_index = np.argsort(preds,1)
    sort_index = np.flip(sort_index)

    for i in range(0, n):
        print(sort_index[0][i], sort_preds[0][i])

    return

def user_test_single_dme(dev_idx):
    """Test single DME."""
    model_file = "../input_models/KL520/mobilenet/models_520.nef"
    model_id = constants.ModelType.IMAGENET_CLASSIFICATION_MOBILENET_V2_224_224_3.value
    image_cfg = (constants.IMAGE_FORMAT_SUB128 | constants.NPU_FORMAT_RGB565 |
                    constants.IMAGE_FORMAT_RAW_OUTPUT | constants.IMAGE_FORMAT_CHANGE_ASPECT_RATIO)

    # load model into Kneron device.
    dme_config = kdp_wrapper.init_dme_config(model_id, 1, image_cfg, ext_param=[0.5])
    ret = kdp_wrapper.dme_load_model(dev_idx, model_file, dme_config)
    if ret == -1:
        return -1

    #get test images ready
    img_path = './data/images/cat.jpg'
    img_path2 = './data/images/fox.jpg'

    npraw_data = kdp_wrapper.dme_inference(dev_idx, 1, img_path)

    # Do postprocessing with keras
    preds = kdp_wrapper.softmax(npraw_data[0]).reshape(1, 1000)
    top_indexes(preds, 3)
    #print('\nPredicted:', decode_predictions(preds, top=3)[0])

    npraw_data = kdp_wrapper.dme_inference(dev_idx, 1, img_path2)

    # Do postprocessing with keras
    preds = kdp_wrapper.softmax(npraw_data[0]).reshape(1, 1000)
    top_indexes(preds, 3)
    #print('\nPredicted:', decode_predictions(preds, top=3)[0])

    return 0

def user_test(dev_idx, _user_id):
    # DME test
    ret = user_test_single_dme(dev_idx)
    kdp_wrapper.end_det(dev_idx)
    return ret
