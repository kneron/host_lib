"""
This is the example for get model info.
"""
from examples_kl520.utils import kdp_wrapper


def user_test_model_info(dev_idx, user_id):
    """User test get model info"""
    print("starting get model info in DDR ...\n")
    ret, model_info = kdp_wrapper.kdp_get_model_info(dev_idx, 1)
    if ret:
        print("Could not get model info..\n")
        return -1

    if (model_info[0] == 0xFFFF):
        print("Not supported by the version of the firmware\n")
    else:
        print("Total model: {}\n".format(model_info[0]))
        for i in range(model_info[0]):
            print("Model {}: {}\n".format(i, model_info[i+1]))

    print("starting get model info in Flash ...\n")
    ret, model_info = kdp_wrapper.kdp_get_model_info(dev_idx, 0)
    if ret:
        print("Could not get model info..\n")
        return -1

    if (model_info[0] == 0xFFFF):
        print("Not supported by the version of the firmware\n")
    else:
        print("Total model: {}\n".format(model_info[0]))
        for i in range(model_info[0]):
            print("Model {}: {}\n".format(i, model_info[i+1]))

    return 0

def user_test(dev_idx, user_id):
    """User test get model info"""
    user_test_model_info(dev_idx, user_id)

    return 0
