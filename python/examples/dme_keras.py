"""
This is the example for dme keras single test.
"""
from python_wrapper import kdp_wrapper
import numpy as np
#from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

def top_indexes(preds, n):
    sort_preds = np.sort(preds,1)
    sort_preds = np.flip(sort_preds)
    sort_index = np.argsort(preds,1)
    sort_index = np.flip(sort_index)

    for i in range(0, n):
        print(sort_index[0][i], sort_preds[0][i])

    return

def user_test_single_dme(dev_idx):
    """Test single dme."""
    # load model into Kneron device
    model_path = "../test_images/dme_mobilenet"
    kdp_wrapper.kdp_dme_load_model(dev_idx, model_path)

    #get test images ready
    img_path = './data/images/cat.jpg'
    img_path2 = './data/images/fox.jpg'

    npraw_data = kdp_wrapper.kdp_inference(dev_idx, img_path)

    # Do postprocessing with keras
    preds = kdp_wrapper.softmax(npraw_data).reshape(1, 1000)
    top_indexes(preds, 3)
    #print('\nPredicted:', decode_predictions(preds, top=3)[0])

    npraw_data = kdp_wrapper.kdp_inference(dev_idx, img_path2)

    # Do postprocessing with keras
    preds = kdp_wrapper.softmax(npraw_data).reshape(1, 1000)
    top_indexes(preds, 3)
    #print('\nPredicted:', decode_predictions(preds, top=3)[0])

    kdp_wrapper.kdp_exit_dme(dev_idx)

def user_test_dme_keras(dev_idx, user_id):
    # dme test
    user_test_single_dme(dev_idx)
    return 0
