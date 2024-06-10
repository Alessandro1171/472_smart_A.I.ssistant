# 472_smart_A.I.ssistant
Group Information:
Yason Bedoshvili – Data Specialist
40058829
 
Alessandro Dare – Training Specialist
40208154
 
Héna Ricucci – Evaluation Specialist
40236544

Contents:
Data Cleaning: the python files containing the code used to clean the images
File(s)

Dataset Visualization: the python files used to construct the data visualization graphs/histograms
File(s):main_dv.py, ClassDistribution.py, data_cleaning.py

Dataset: A text file detailing the information of each dataset we sourced from
File(s):Datasets.txt

Archive: A zip file containing an archive of 100 sample images (25 of each class) for our complete dataset
Files(S): image_archive

Report: The report containing our mythology, processes, findings, analysis and procedures for project part1
File(s):SmartClass A.I.ssistant

Originality Form: Documents that confirm all out work is our own and all references and clearly defined
File(s):Originality_form_alessandro_dare.pdf, Originality_form_Hena_Ricucci.pdf, Expectations-of-Originality-Yason-Bedoshvili.pdf

Data Cleaning Instructions:
The paths for the directories are our local paths, please make sure that you replace them with the paths where the folders are located.

First file is preprocess_images.py
1. Make sure you have the following dependencies installed:
os, 
PIL

2. Make sure that directories have the correct paths
3. After execution of the code, please, put images in the different folders according to the emotions on the face

Second file is CreateTrainTest.py
1. Make sure you have the following dependencies installed:
os, 
shutil, 
random

2. Puth the right paths in the code for angry, happy, and neutral

Third file is DatasetLoading.py
1. Make sure you have the following dependencies installed:
os, 
PIL, 
torch, 
numphy, 
torchvision

2. Puth the right paths for all emotions of test and train

Fourth file is Preprocessing.py
1. Make sure you have the following dependencies installed:
os, 
PIL, 
hashlib

2. Puth the right paths for all emotions of test and train folders


Data Visualization Instructions:
1. Make sure you have the following dependencies installed:
matplotlib, 
cv2 (opencv-python), 
numpy, 
tensorflow, 
glob, 
random, 
PIL
 
2. Make sure that the files: ClassDistribution.py, main_dv.py, SampleImages.py are in the same folder/driectorty.

3. In main.py file there are two variables: paths_train and paths_test
These contain the paths to the directories that contain the train images for each class and the test images for each class respectively.
Replace the current paths with your local paths.

4. Run the main_dv.py file
This will cause the class distribution graph, the pixel distribution graph, and the Sample Grid, for each class in that order.
Each visualization section has its own figure, just X out of the current figure to move to the next one.
Note: the pixel distribution graph could take a long time to load (over 5 min) so don't try to restart the program if it seems to be taking a while.

If you wish to skip certain figure just comment them out of the main.py file (lines that call which figure are commented).
