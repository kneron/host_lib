# Kneron Host Lib

This project contains python examples for Kneron USB dongle

## Install Host Lib Python Package

### Prerequisites

**General**:

* Python 3.8
* opencv-python (Install using `pip install opencv-python`)

**Linux**:

* libusb (Ubuntu install using `apt install libusb-1.0-0-dev`)

**Windows**:

* WinUSB (Install after plugging in the dongle and run [Zadig](https://zadig.akeo.ie/))

**Mac OS**:

* libusb (Install with [Homebrew](https://brew.sh/) using `brew install libusb`)

### Installation

Simply run the following command but replace the item within `<>` .

```bash
pip install pkgs/kdp_host_api-<version>_<os_version>_-py3-none-any.whl
```

`os_version` should be `win`, `mac` or`linux`.

Please check the `pkgs` folder for the `version` available.

### Troubleshooting

**If the size of the `.whl` is 0, please use `git-lfs pull` to fetch the pacakge.***

## Run examples

### Getting started with help

1. **Please make sure you have installed the packages mentioned in the previous section.**
2. Plug in the dongle.
3. Using command line tool and enter the `python` folder.
4. Run `python main.py -h`

If all of the packages are installed correctly, you would get a message with available examples like the following:

```
Run Python examples by calling the Python APIs

optional arguments:
  -h, --help            show this help message and exit
  -t TASK_NAME, --task_name TASK_NAME
                        cam_dme_ssd_fd
                        cam_yolo
                        cam_isi_ssd_fd
                        cam_isi_yolo
                        cam_isi_parallel_yolo
                        dme_keras
                        update_app
                        update_fw
```

### Run Examples

There are two kinds of examples, need `update_app` and don't need extra steps.

#### No Extra Step Example

Let's first start with the example do not need extra steps: `dme_keras`.

You just need to run the command `python main.py -t dme_keras` with your dongle plugged in. Then, you can get the output without any error message.

```bash
$ python main.py -t dme_keras
adding devices....

start kdp host lib....

Task:  dme_keras
loading models to Kneron Device:
starting DME mode ...

DME mode succeeded...

Model loading successful
starting DME configure ...

DME configure model [1000] succeeded...

./data/images/cat.jpg
281 0.513678242399822
285 0.23041174139774295
282 0.15431613018623724
./data/images/fox.jpg
277 0.8318471411560402
278 0.06568296365719047
272 0.057467412876039244
de init kdp host lib....
```

#### Examples need `update_app`

Most examples need extra steps to upload the model file into the dongle. You can find the model files under `app_binaries`. Here is a table of the relationship between models and examples:

| Example                | Model        |
|------------------------|--------------|
| cam_dme_ssd_fd         | ssd_fd       |
| cam_isi_ssd_fd         | ssd_fd       |
| cam_yolo               | tiny_yolo_v3 |
| cam_isi_yolo           | tiny_yolo_v3 |
| cam_isi_parallel_yolo  | tiny_yolo_v3 |

Here are the steps you need to update the dongle and run the example. Let's take `cam_isi_yolo` as the example.

1. From the chart, we know we need to use model `tiny_yolo_v3`.
2. Copy the binary files (`*.bin`) under `app_binaries/tiny_yolo_v3` into `ota/ready_to_load`.
3. Enter `python` directory.
4. Run `python main.py -t update_app` (This step may take some time).
5. Run `python main.py -t cam_isi_yolo`.

Now, you can get a window pop up and running your test. Congratulations!

To stop the test and quit, just press `q` when focusing on the command line.
