"""
This is the 720 get NEF metadata example.
"""
import pathlib

from common import kdp_wrapper

# Example model/image paths
HOST_LIB_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
TEST_OTA_DIR = HOST_LIB_DIR / "app_binaries/KL720/dfu/ready_to_load"
MODEL_FILE = str(TEST_OTA_DIR / "models_720.nef")

def user_test_metadata():
    """Gets the metadata from the input NEF model."""
    print("\nStart getting metadata from NEF model file...")

    metadata = kdp_wrapper.get_nef_model_metadata(MODEL_FILE)
    if metadata is None:
        return -1

    print(metadata)
    return 0

def user_test(_device_index, _user_id):
    """Get metadata test."""
    return user_test_metadata()
