"""
This is the example for soft reset.
"""
from common import kdp_wrapper

SOFT_RESET = 255

def user_test_reset(device_index, user_id):
    """User test soft reset."""
    print("\nStarting soft reset...")

    return kdp_wrapper.reset_sys(device_index, SOFT_RESET)

def user_test(device_index, user_id):
    """User test soft reset."""
    user_test_reset(device_index, user_id)

    return 0
