"""
This is the example for getting metadata from NEF model file.
"""
import pathlib

from common import kdp_wrapper

HOST_LIB_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
TEST_OTA_DIR = HOST_LIB_DIR / "app_binaries/KL520/ota/ready_to_load"
MODEL_FILE = str(TEST_OTA_DIR / "models_520.nef")

def user_test_metadata(device_index, user_id):
    """User test get metadata"""
    print("\nStart getting metadata from NEF model file...")

    metadata = kdp_wrapper.get_nef_model_metadata(MODEL_FILE)
    if metadata is None:
        return -1

    print(metadata)
    return 0

def user_test(device_index, user_id):
    """User test get metadata."""
    user_test_metadata(device_index, user_id)

    return 0
