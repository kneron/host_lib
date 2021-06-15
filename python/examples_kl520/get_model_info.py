"""
This is the example for getting model info.
"""
from common import kdp_wrapper

def user_test_model_info(device_index, user_id):
    """User test get model info"""
    print("\nStart getting model info in DDR...")

    model_info = kdp_wrapper.get_model_info(device_index, 1)

    if not model_info:
        return -1
    elif model_info[0] == 0xFFFF:
        print("Not supported by the version of the firmware")
    else:
        print(f"\nTotal model: {model_info[0]}")
        for index, model_num in enumerate(model_info[1:]):
            print(f"Model {index}: {model_num}")

    print("\nStart getting model info in Flash...")
    model_info = kdp_wrapper.get_model_info(device_index, 0)

    if not model_info:
        return -1
    elif model_info[0] == 0xFFFF:
        print("Not supported by the version of the firmware")
    else:
        print(f"\nTotal model: {model_info[0]}")
        for index, model_num in enumerate(model_info[1:]):
            print(f"Model {index}: {model_num}")

    return 0

def user_test(device_index, user_id):
    """User test get model info."""
    user_test_model_info(device_index, user_id)

    return 0
