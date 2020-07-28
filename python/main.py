"""
This is the main function to run any of the Python examples.
"""
import argparse
from examples.cam_isi_ssd_fd import user_test_cam_isi_ssd_fd
from examples.cam_yolo import user_test_cam_yolo
from examples.cam_isi_yolo import user_test_cam_isi_yolo
from examples.cam_isi_parallel_yolo import user_test_cam_isi_parallel_yolo
from examples.cam_dme_ssd_fd import user_test_cam_dme_ssd_fd
from examples.dme_keras import user_test_dme_keras
from examples.update_app import user_test_update_app
from examples.update_fw import user_test_update_fw
from kdp_host_api import (
    kdp_add_dev, kdp_init_log, kdp_lib_de_init, kdp_lib_init, kdp_lib_start)

KDP_UART_DEV = 0
KDP_USB_DEV = 1

if __name__ == "__main__":
    ### input parameters ###
    argparser = argparse.ArgumentParser(
        description="Run Python examples by calling the Python APIs",
        formatter_class=argparse.RawTextHelpFormatter)

    argparser.add_argument(
        '-t',
        '--task_name',
        help=("cam_dme_ssd_fd\ncam_yolo\ncam_isi_ssd_fd\ncam_isi_yolo\ncam_isi_parallel_yolo\n"
              "dme_keras\nupdate_app\nupdate_fw"))

    args = argparser.parse_args()

    ### initialize Kneron USB device ###
    kdp_init_log("/tmp/", "mzt.log")

    if kdp_lib_init() < 0:
        print("init for kdp host lib failed.\n")

    print("adding devices....\n")

    dev_idx = kdp_add_dev(KDP_USB_DEV, "")
    if dev_idx < 0:
        print("add device failed.\n")

    print("start kdp host lib....\n")
    if kdp_lib_start() < 0:
        print("start kdp host lib failed.\n")

    user_id = 0

    print("Task: ", args.task_name)
    ### parse parameters and run different example ###
    {
        "cam_isi_ssd_fd": user_test_cam_isi_ssd_fd,
        "cam_yolo": user_test_cam_yolo,
        "cam_isi_yolo": user_test_cam_isi_yolo,
        "cam_isi_parallel_yolo": user_test_cam_isi_parallel_yolo,
        "cam_dme_ssd_fd": user_test_cam_dme_ssd_fd,
        "dme_keras": user_test_dme_keras,
        "update_app": user_test_update_app,
        "update_fw": user_test_update_fw,
    }.get(args.task_name, lambda: 'Invalid test')(dev_idx, user_id)

    ### de-initialize Kneron USB device ###
    print("de init kdp host lib....\n")
    kdp_lib_de_init()
