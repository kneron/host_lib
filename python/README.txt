How to test Python examples with Python3

1. Install Kneron Host Side API with Python whl file

2. Install opencv
pip3 install opencv-python

3. Help
cd python
python main.py -h

4. Update firmware for Kneron device if needed
(1) Copy the app binaries (*.bin) in app_binaries into app_binaries/ota/ready_to_load
(2) python main.py -t update_app

5. Example
python main.py -t dme_keras