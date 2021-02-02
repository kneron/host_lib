"""
This is the example for update app with nef model.
"""
import ctypes
import kdp_host_api as api

HOST_LIB_DIR = ""
TEST_OTA_DIR = "".join([HOST_LIB_DIR, "../app_binaries/KL520/ota"])
FW_SCPU_FILE = "".join([TEST_OTA_DIR, "/ready_to_load/fw_scpu.bin"])
FW_NCPU_FILE = "".join([TEST_OTA_DIR, "/ready_to_load/fw_ncpu.bin"])
MODEL_FILE = "".join([TEST_OTA_DIR, "/ready_to_load/models_520.nef"])

FW_FILE_SIZE = 128 * 1024
MD_FILE_SIZE = 20 * 1024 * 1024

def user_test_app(dev_idx, user_id):
    """User test update firmware"""
    # udt firmware
    module_id = user_id
    img_buf = (ctypes.c_char * FW_FILE_SIZE)()

    if module_id not in (0, 1, 2):
        print("invalid module id: {}...\n".format(user_id))
        return -1

    print("starting update fw ...\n")

    # update scpu
    module_id = 1
    # print(FW_SCPU_FILE)
    buf_len_ret = api.read_file_to_buf(img_buf, FW_SCPU_FILE, FW_FILE_SIZE)

    if buf_len_ret <= 0:
        print("reading file to buf failed: {}...\n".format(buf_len_ret))
        return -1

    buf_len = buf_len_ret
    #print("buf len is ", buf_len)
    ret, module_id = api.kdp_update_fw(dev_idx, module_id, img_buf, buf_len)
    if ret:
        print("could not update fw..\n", ret)
        return -1

    print("update SCPU firmware succeeded...\n")

    # update ncpu
    buf_len_ret = api.read_file_to_buf(img_buf, FW_NCPU_FILE, FW_FILE_SIZE)
    module_id = 2
    if buf_len_ret <= 0:
        print("reading file to buf failed: {}...\n".format(buf_len_ret))
        return -1

    buf_len = buf_len_ret
    ret, module_id = api.kdp_update_fw(dev_idx, module_id, img_buf, buf_len)
    if ret:
        print("could not update fw..\n", ret)
        return -1

    print("update NCPU firmware succeeded...\n")

    # update model
    p_buf = (ctypes.c_char * MD_FILE_SIZE)()

    print("starting update model...\n")

    buf_len_ret = api.read_file_to_buf(p_buf, MODEL_FILE, MD_FILE_SIZE)
    if buf_len_ret <= 0:
        print("reading file to buf failed: {}...\n".format(buf_len_ret))
        return -1

    buf_len = buf_len_ret

    ret = api.kdp_update_nef_model(dev_idx, p_buf, buf_len)
    if ret:
        print("could not update model..\n")
        return -1

    print("update model succeeded...\n")
    return 0

def user_fw_id(dev_idx):
    """User test get version ID"""
    print("starting report sys status ...\n")
    ret, sfirmware_id, sbuild_id, _sys_status, _app_status, nfirmware_id, nbuild_id = (
        api.kdp_report_sys_status(dev_idx, 0, 0, 0, 0, 0, 0))
    if ret:
        print("could not report sys status..\n")
        return -1

    print("report sys status succeeded...\n")
    print("\nFW firmware_id {}.{}.{}.{} build_id {}\n".format(
        sfirmware_id >> 24, (sfirmware_id & 0x00ff0000) >> 16,
        (sfirmware_id & 0x0000ff00) >> 8,
        (sfirmware_id & 0x000000ff), sbuild_id))
    return 0

def user_test(dev_idx, user_id):
    """User test update app"""
    # udt application test
    user_test_app(dev_idx, user_id)
    # udt application id test
    user_fw_id(dev_idx)

    return 0
