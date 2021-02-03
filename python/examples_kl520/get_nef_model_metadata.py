"""
This is the example for get metadata from nef model file.
"""
from examples_kl520.utils import kdp_wrapper

HOST_LIB_DIR = ""
TEST_OTA_DIR = "".join([HOST_LIB_DIR, "../app_binaries/KL520/ota"])
MODEL_FILE = "".join([TEST_OTA_DIR, "/ready_to_load/models_520.nef"])

def user_test_metadata(dev_idx, user_id):
    """User test get metadata"""
    print("starting get metadata from nef model file ...\n")
    ret, metadata = kdp_wrapper.kdp_get_nef_model_metadata(MODEL_FILE)
    if ret:
        print("Could not get metadata..\n")
        return -1

    print("\n")
    print("platform:", str(metadata[0], encoding='utf-8'))
    print("target: {}".format(metadata[1]))  # 0: KL520, 1: KL720, etc.
    print("crc: 0x{:06x}".format(metadata[2]))
    print("kn number: {:06x}".format(metadata[3]))
    print("encrypt type: {}".format(metadata[4]))
    print("toolchain version:", str(metadata[5], encoding='utf-8'))
    print("compiler version:", str(metadata[6], encoding='utf-8'))
    print("\n")

    return 0

def user_test(dev_idx, user_id):
    """User test get metadata"""
    user_test_metadata(dev_idx, user_id)

    return 0
