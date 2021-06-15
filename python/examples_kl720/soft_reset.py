"""
This is the 720 soft reset example.
"""
from common import kdp_wrapper

# Predefined soft reset code.
SOFT_RESET = 255

def user_test_reset(device_index):
    """User test soft reset."""
    print("\nStarting soft reset...")

    return kdp_wrapper.reset_sys(device_index, SOFT_RESET)

def user_test(device_index, _user_id):
    """Soft reset."""
    return user_test_reset(device_index)
