"""
This is the example for get crc.
"""
from common import kdp_wrapper

def user_test_crc(device_index, _user_id):
    """User test get CRC."""
    print("\nStart getting CRC in DDR...")

    ddr = 1
    flash = 0

    crc = kdp_wrapper.get_crc(device_index, ddr)

    if crc == -1:
        return -1
    elif crc != -2:
        print("CRC: {:06x}\n".format(crc))

    print("\nStart getting CRC in Flash...")
    crc = kdp_wrapper.get_crc(device_index, flash)

    if crc == -1:
        return -1
    elif crc != -2:
        print("CRC: 0x{:06x}\n".format(crc))

    return 0

def user_test(device_index, user_id):
    """User test get CRC."""
    user_test_crc(device_index, user_id)

    return 0
