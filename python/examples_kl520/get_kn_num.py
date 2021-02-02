"""
This is the example for get kn number.
"""
from examples_kl520.utils import kdp_wrapper


def user_test_kn_num(dev_idx, user_id):
    """User test get kn number"""
    print("starting get KN number ...\n")
    ret, kn_number = kdp_wrapper.kdp_get_kn_number(dev_idx, 0)
    if ret:
        print("Could not get KN number..\n")
        return -1

    if (kn_number == 0xFFFF):
        print("Not supported by the version of the firmware\n")
    else:
        print("KN number: 0x{:06x}\n".format(kn_number))

    return 0

def user_test(dev_idx, user_id):
    """User test get kn number"""
    user_test_kn_num(dev_idx, user_id)

    return 0
