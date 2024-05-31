# 472_smart_A.I.ssistant

Data Visualization Instructions:
1. Make sure you have the following dependencies installed:
matplotlib
cv2 (opencv-python)
numpy
tensorflow
glob
random
PIL
 
2.Make sure that the files: ClassDistribution.py, main_dv.py, SampleImages.py are in the same folder/driectorty.

3.In main.py file there are two variables: paths_train and paths_test
These contain the paths to the directories that contain the train images for each class and the test images for each class respectively.
Replace the current paths with your local paths.

4.Run the main_dv.py file
This will cause the class distribution graph, the pixel distribution graph, and the Sample Grid, for each class in that order.
Each visualization section has its own figure, just X out of the current figure to move to the next one.
Note: the pixel distribution graph could take a long time to load (over 5 min) so don't try to restart the program if it seems to be taking a while.

If you wish to skip certain figure just comment them out of the main.py file (lines that call which figure are commented).
