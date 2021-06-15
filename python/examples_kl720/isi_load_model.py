"""
This is the 720 ISI load model example.
"""
import pathlib
import time

from common import constants, kdp_wrapper

# Example model/image paths
ROOT_FOLDER = pathlib.Path(__file__).resolve().parent.parent.parent
MODEL_FOLDER = ROOT_FOLDER / "app_binaries/KL720/dfu/ready_to_load"
MODEL_FILE = str(MODEL_FOLDER / "models_720.nef")

# App ID of APP_CENTER_APP will always works for a single model
ISI_APP_ID = constants.AppID.APP_CENTER_APP.value

MAX_MODEL_SIZE = constants.MAX_MODEL_SIZE_720
ISI_START_DATA_SIZE = constants.ISI_START_DATA_SIZE_720

def load_model_file(device_index):
    """Load the NEF model file onto the board."""
    print("\nStart loading model...")

    return kdp_wrapper.isi_load_nef(device_index, MODEL_FILE, ISI_APP_ID)

def user_test(device_index, _user_id):
    """Runs example of loading a model to the board."""
    start_time = time.time()

    ret = load_model_file(device_index)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"=> Time = {elapsed_time:.3f} seconds\n")

    return ret
