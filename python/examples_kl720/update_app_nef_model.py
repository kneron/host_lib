"""
This is the 720 update firmware and NEF model example.
"""
import pathlib

from common import constants, kdp_wrapper

# Example model/image paths
HOST_LIB_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
TEST_OTA_DIR = HOST_LIB_DIR / "app_binaries/KL720/dfu/ready_to_load"
FW_SCPU_FILE = str(TEST_OTA_DIR / "fw_scpu.bin")
FW_NCPU_FILE = str(TEST_OTA_DIR / "fw_ncpu.bin")
MODEL_FILE = str(TEST_OTA_DIR / "models_720.nef")

FW_FILE_SIZE = constants.FW_FILE_SIZE_720
MODEL_FILE_SIZE = constants.MAX_MODEL_SIZE_720
SCPU_ID = 1
NCPU_ID = 2

def user_test_app(device_index):
    """User test update firmware."""
    print("\nStart updating fw...")

    # Update SCPU
    if kdp_wrapper.update_fw(device_index, SCPU_ID, FW_SCPU_FILE, FW_FILE_SIZE):
        return -1

    # Update NCPU
    if kdp_wrapper.update_fw(device_index, NCPU_ID, FW_NCPU_FILE, FW_FILE_SIZE):
        return -1

    print("\nStart updating model...")
    # Update model
    if kdp_wrapper.update_nef_model(device_index, MODEL_FILE, MODEL_FILE_SIZE):
        return -1

    return 0

def user_fw_id(device_index):
    """Get system status."""
    print("\nStart reporting sys status...")

    return kdp_wrapper.report_sys_status(device_index)

def user_test(device_index, _user_id):
    """Update firmware and NEF test."""
    ret = user_test_app(device_index)
    if ret:
        return -1

    ret = user_fw_id(device_index)
    return ret
