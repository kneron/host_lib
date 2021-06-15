"""
This is the example for get kn number.
"""
from common import kdp_wrapper

def user_test_kn_num(device_index, user_id):
    """User test get kn number."""
    print("\nStart getting KN number...")

    kn_number = kdp_wrapper.get_kn_number(device_index)

    if kn_number == -1:
        return -1
    elif kn_number != -2:
        print(f"\nKN number: 0x{kn_number:06x}\n")

    return 0

def user_test(device_index, user_id):
    """User test get kn number."""
    user_test_kn_num(device_index, user_id)

    return 0
