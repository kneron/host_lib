"""
This is the main function to run any of the Python examples_kl520.
"""
import argparse
import pkgutil
import importlib
import os
import re
from common import kdp_wrapper

''' kdp_host_api config '''
KDP_UART_DEV = 0
KDP_USB_DEV = 1
VENDOR_ID = 0x3231
KL520_PRODUCT_ID = 0x0100
KL720_PRODUCT_ID = 0x0200

''' example import config '''
EXAMPLE_FUNCTION_NAME = 'user_test'

''' import ignore list '''
IGNORE_MODULE_LIST = [
    'utils',
    'fdssd',
    'yolo',
    'keras_only',
    'fdr'
]


def get_module_names(examples_dir, example_regex=r'^examples_kl\d{3}'):
    module_names = []
    folder_names = [name for name in os.listdir(examples_dir) if os.path.isdir(name)]
    for folder_name in folder_names:
        if re.match(pattern=example_regex, string=folder_name):
            module_names.append(folder_name)
    return module_names


def get_all_module_path(package_path_list):
    example_dict = {}

    for package_path in package_path_list:
        for finder, name, _ in pkgutil.iter_modules(path=[package_path]):
            package_name = os.path.basename(finder.path)
            search_obj = re.search(pattern=r'(kl\d\d\d)$', string=package_name)

            if search_obj and (name not in IGNORE_MODULE_LIST):
                kneron_device_name = search_obj.group(1)
                example_dict['-'.join([kneron_device_name.upper(), name])] = '{}.{}'.format(package_name, name)
    return example_dict


def import_example_function(module_path, function_name):
    try:
        module = importlib.import_module(module_path)
        if hasattr(module, function_name):
            return getattr(module, function_name)
    except Exception as e:
        print('Can not import {}, function {}'.format(module_path, function_name))


''' get example modules '''
example_module_folder_path = os.path.dirname(os.path.abspath(__file__))
module_names = get_module_names(examples_dir=example_module_folder_path,
                                example_regex=r'^examples_kl\d\d\d$')
example_dict = get_all_module_path(
    package_path_list=[os.path.join(example_module_folder_path, module_name) for module_name in module_names])

if __name__ == "__main__":
    ''' input parameters '''
    argparser = argparse.ArgumentParser(
        description="Run Python examples by calling the Python APIs",
        formatter_class=argparse.RawTextHelpFormatter)

    argparser.add_argument(
        '-t',
        '--task_name',
        help=('\n'.join(example_dict.keys())))

    argparser.add_argument(
        '-p',
        '--param_list',
        help=('Any input params to pass to the test you run'),
        nargs='*'
    )

    args = argparser.parse_args()

    '''  initialize Kneron USB device '''
    kdp_wrapper.init_log("/tmp/", "mzt.log")

    if kdp_wrapper.lib_init() < 0:
        print("init for kdp host lib failed.\n")
        exit(-1)

    print("adding devices....\n")

    category = args.task_name[:5]
    if category == "KL520":
        pid = KL520_PRODUCT_ID
    elif category == "KL720":
        pid = KL720_PRODUCT_ID
    else:
        print("Task category not in KL520 or KL720: ", category)
        exit(-1)

    dev_idx = -1

    # dev_list is the list of [scan_index, isConnectable, vendor_id, product_id, link_speed, serial_number, device_path]
    ret, dev_list = kdp_wrapper.scan_usb_devices()

    for i in range(len(dev_list)):
        if (dev_list[i][3] == pid):#get the first KL520 or KL720 device
            # found it, now try to connect
            dev_idx = kdp_wrapper.connect_usb_device(dev_list[i][0])
            break

    # print("dev_idx:", dev_idx)

    if dev_idx < 0:
        print("add device failed.\n")
        exit(-1)

    print("start kdp host lib....\n")
    if kdp_wrapper.lib_start() < 0:
        print("start kdp host lib failed.\n")
        exit(-1)

    user_id = 0

    print("Task: ", args.task_name)
    '''  parse parameters and run different example '''
    example_function = import_example_function(module_path=example_dict[args.task_name],
                                               function_name=EXAMPLE_FUNCTION_NAME)

    if args.param_list is not None:
        example_function(dev_idx, user_id, args.param_list)
    else:
        example_function(dev_idx, user_id)

    '''  de-initialize Kneron USB device '''
    print("de init kdp host lib....\n")
    kdp_wrapper.lib_de_init()
