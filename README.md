This script labels ten meter walk and run tests with mediapipe!

To use it, download this folder and install the necessary libraries by opening a terminal window, navigating to the folder that you've downloaded, and running these three lines:

python3 -m venv .venv                ## this sets up a local python environment for us to install our needed libraries into

source .venv/bin/activate            ## this switches us to the new environment so when we install we install locally

pip install requirements.txt         ## this will install all the necessary packages

After, you call the script like so:

python3 label_aruco.py <input directory> --output_dir <output directory>

In this call, <input directory> contains all of the video files you want to label and <output directory> tells the script where to store the 3D point results.
