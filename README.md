## This script labels ten meter walk and run tests with mediapipe!

To use it, download the project folder and install the necessary libraries by opening a terminal window, navigating to the folder that you've downloaded, and running these three lines:

#### *set up a local python environment for us to install our needed libraries into*
```
python3 -m venv .venv
```

#### *switch to the new environment so when we install we install locally*
```
source .venv/bin/activate
```

#### *install all the necessary packages*
```
pip install requirements.txt
```

#### After, you call the script like so:
```
python3 label_aruco.py --input_dir <input directory> --output_dir <output directory>
```

In this call, ```<input directory>``` contains all of the video files you want to label and ```<output directory>``` tells the script where to store the 3D point results.
