"""
This is the 720 get model info example.
"""
from common import kdp_wrapper

DDR_ID = 1
FLASH_ID = 0

def user_test_model_info(device_index):
    """Get model info from flash and DDR on the board."""
    print("\nStart getting model info in DDR...")
    model_info = kdp_wrapper.get_model_info(device_index, DDR_ID)
    if not model_info:
        return -1

    if model_info[0] == 0xFFFF:
        print("Not supported by the version of the firmware")
    else:
        print(f"\nTotal model: {model_info[0]}")
        for index, model_num in enumerate(model_info[1:]):
            print(f"Model {index}: {model_num}")

    print("\nStart getting model info in Flash...")
    model_info = kdp_wrapper.get_model_info(device_index, FLASH_ID)
    if not model_info:
        return -1

    if model_info[0] == 0xFFFF:
        print("Not supported by the version of the firmware")
    else:
        print(f"\nTotal model: {model_info[0]}")
        for index, model_num in enumerate(model_info[1:]):
            print(f"Model {index}: {model_num}")

    return 0

def user_test(device_index, _user_id):
    """Get model info test."""
    return user_test_model_info(device_index)
