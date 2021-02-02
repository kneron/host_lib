"""
This is the example for get crc.
"""
from examples_kl520.utils import kdp_wrapper


def user_test_crc(dev_idx, user_id):
    """User test get crc"""
    print("starting get crc in DDR ...\n")
    user_id = 1
    ret, crc = kdp_wrapper.kdp_get_crc(dev_idx, user_id)
    if ret:
        print("Could not get crc..\n")
        return -1

    if (crc == 0xFFFF):
        print("Not supported by the version of the firmware\n")
    else:
        if (crc == 0 and user_id):
            print("Models have not been loaded into DDR\n")
        elif (crc == 0xFFFFFFFF):
            print("No CRC info for the loaded models\n")
        else:
            print("CRC: {:06x}\n".format(crc))

    print("starting get crc in Flash ...\n")
    user_id = 0
    ret, crc = kdp_wrapper.kdp_get_crc(dev_idx, user_id)
    if ret:
        print("Could not get crc..\n")
        return -1

    if (crc == 0xFFFF):
        print("Not supported by the version of the firmware\n")
    elif (crc == 0xFFFFFFFF):
        print("No CRC info for the loaded models\n")
    else:
        print("CRC: 0x{:06x}\n".format(crc))

    return 0

def user_test(dev_idx, user_id):
    """User test get crc"""
    user_test_crc(dev_idx, user_id)

    return 0
