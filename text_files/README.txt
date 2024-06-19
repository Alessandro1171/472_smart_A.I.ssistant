472_smart_A.I.ssistant
Githib:https://github.com/Alessandro1171/472_smart_A.I.ssistant
Group Information: Yason Bedoshvili – Data Specialist 40058829

Alessandro Dare – Training Specialist 40208154

Héna Ricucci – Evaluation Specialist 40236544



Contents: Data Cleaning: the python files containing the code used to clean the images 
File(s):preprocess_images.py, CreateTrainTest.py, DatasetLoading.py,Preprocessing.py

Dataset Visualization: the python files used to construct the data visualization graphs/histograms 
File(s):main_dv.py, ClassDistribution.py, data_cleaning.py

CNN Architecture: the python files used to set up the dataset, store the models and train the model
File(s):trained_ai.py, dataset_split.py,CNN_Image_Scanner_V1.py,CNN_Image_Scanner_V2.py,CNN_Image_Scanner_V3.py
CNN Models are stored in CNN_Image_Scanner_V1.py,CNN_Image_Scanner_V2.py,CNN_Image_Scanner_V3.py

Evaltuion:the python files used to evaluate the best model variants
File(s):load_Evaluate.py

Dataset: A text file detailing the information of each dataset we sourced from 
File(s):Datasets.txt

Archive: A zip file containing an archive of 100 sample images (25 of each class) for our complete dataset 
Files(S): image_archive

Report: The report containing our mythology, processes, findings, analysis and procedures for project part1 
File(s):SmartClass A.I.ssistant

Originality Form: Documents that confirm all out work is our own and all references and clearly defined File(s):Originality_form_alessandro_dare.pdf, Originality_form_Hena_Ricucci.pdf, Expectations-of-Originality-Yason-Bedoshvili.pdf



Data Cleaning Instructions: The paths for the directories are our local paths, please make sure that you replace them with the paths where the folders are located.

First file is preprocess_images.py

Make sure you have the following dependencies installed: os, PIL

Make sure that directories have the correct paths

After execution of the code, please, put images in the different folders according to the emotions on the face

Second file is CreateTrainTest.py

Make sure you have the following dependencies installed: os, shutil, random

Puth the right paths in the code for angry, happy, and neutral

Third file is DatasetLoading.py

Make sure you have the following dependencies installed: os, PIL, torch, numphy, torchvision

Puth the right paths for all emotions of test and train

Fourth file is Preprocessing.py

Make sure you have the following dependencies installed: os, PIL, hashlib

Puth the right paths for all emotions of test and train folders

Data Visualization Instructions:

Make sure you have the following dependencies installed: matplotlib, cv2 (opencv-python), numpy, tensorflow, glob, random, PIL

Make sure that the files: ClassDistribution.py, main_dv.py, SampleImages.py are in the same folder/driectorty.

In main.py file there are two variables: paths_train and paths_test These contain the paths to the directories that contain the train images for each class and the test images for each class respectively. Replace the current paths with your local paths.

Run the main_dv.py file This will cause the class distribution graph, the pixel distribution graph, and the Sample Grid, for each class in that order. Each visualization section has its own figure, just X out of the current figure to move to the next one. Note: the pixel distribution graph could take a long time to load (over 5 min) so don't try to restart the program if it seems to be taking a while.

If you wish to skip certain figure just comment them out of the main.py file (lines that call which figure are commented).

Running Model instructions:The paths for the directories are our local paths, please make sure that you replace them with the paths where the data folders are located. make sure  files trained_ai.py, load_Evaluate.py,CNN_Image_Scanner_V1.py, CNN_Image_Scanner_V2.py,  CNN_Image_Scanner_V3.py, are in the same directory

First File:dataset_split.py
specify the base paths for the directories of each class in the base_paths dictionary that you want them to be stored make sure all classes share the same directory
run the file

Second file:trained_ai.py (needed:CNN_Image_Scanner_V1.py,CNN_Image_Scanner_V2.py,CNN_Image_Scanner_V3.py)
In the Pclass specify the path variable in the directory where all the classes are stored
Run the file 3 times
At line 312 change the CNN_Image_Scanner class from CNN_Image_Scanner_V1, CNN_Image_Scanner_V2, CNN_Image_Scanner_V3 for every run and 
at line 275 torch.save(model.state_dict(), 'best_model_v1.pth') change best_model to best_model_v1, best_model_v2, best_model_v3 with every run

Thrid file:load_Evaluate.py
Run the file three times with in line 200, and 203 change model from CNN_Image_Scanner_V1, best_model_v1.pth to CNN_Image_Scanner_V2, best_model_v2.pth and CNN_Image_Scanner_V3, best_model_v3.pth, make sure to capture and record the results