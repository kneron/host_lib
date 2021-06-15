"""
This is the 720 update firmware example.
"""
import pathlib

from common import constants, kdp_wrapper

# Example model/image paths
HOST_LIB_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
TEST_OTA_DIR = HOST_LIB_DIR / "app_binaries/KL720/dfu/ready_to_load"
FW_SCPU_FILE = str(TEST_OTA_DIR / "fw_scpu.bin")
FW_NCPU_FILE = str(TEST_OTA_DIR / "fw_ncpu.bin")

FW_FILE_SIZE = constants.FW_FILE_SIZE_720
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

    return 0

def user_fw_id(device_index):
    """Get system status."""
    print("\nStart reporting sys status...")

    return kdp_wrapper.report_sys_status(device_index)

def user_test(device_index, _user_id):
    """Update firmware test."""
    ret = user_test_app(device_index)
    if ret:
        return -1

    ret = user_fw_id(device_index)
    return ret
