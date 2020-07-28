# Host Lib Document for Python Packages

## Prerequisites

### General

* Python 3.8
* opencv-python (Install using `pip install opencv-python`)

### Linux

* libusb (Ubuntu install using `apt install libusb-1.0-0-dev`)

### Windows

* WinUSB (Install after plugging in the dongle and run [Zadig](https://zadig.akeo.ie/))

### Mac OS

* libusb (Install with [Homebrew](https://brew.sh/) using `brew install libusb`)

## Installing

`pip install pkgs/kdp_host_api-<version>_<os_version>_-py3-none-any.whl`

`os_version` should be `win`, `mac` or`linux`.

Please check the `pkgs` folder for details.

## Usage

After installation, just `import kdp_host_api` in Python and start coding!