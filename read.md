# face_mask_detection

We all have faced the harsh effect of COVID-19 in our lives and have realized the true need of masks. But one of things that was still missing is the proper monitoring system. We might not be able to know whether people have wore masks and it is also not possible for people to keep track of it. That is not a reliable solution and has to to be automated.
To solve this problem, we will take the help of Machine learning.

This project utilizes some of the best face detection algorithm/models out there and uses that to identify people faces with or without masks.
We will be preprocessing and train our models with the images that we have for our dataset and use that to identify the images and also live stream for mask detection.
The indepth code explaination is done in the script files so, please make sure to take a look. You can run the ".py" file in the cmd or can run the ".ipynb" file in jupyter to see the step by step process.

**Setting up the virtual environment**

We will create the virtual environment to make sure that we have all the required version of the packages for running the code without errors.
To create a virtual environment, decide upon a directory where you want to place it, and run the venv module as a script with the directory path:

```
python3 -m venv your_venv_name
```

Once you’ve created a virtual environment, you may activate it.

On Windows, run:

```
your_venv_name\Scripts\activate.bat
```

On Unix or MacOS, run:

```
source your_venv_name/bin/activate
```

**Installation of dependencies**

Before running anything after cloning the repository, please make sure to install the required packages using command:

```
pip install -r requirements.txt
```

**For detecting the image run "detect_mask_image.py" with command**

```
python detect_mask_image.py -i <location to a test image,required> <path to face detection model,default = 'face_detector',optional>  <path to mask detection model,default = 'mask_detector.model',optional> <minimum probability to filter the weak detection, default = 0.5, optional>

Example: "python detect_mask_image -i images\00025.png"
```

**_To detect the images in jupyter(.ipynb extension) instead of running the script directly, open a jupyter session and open "detect_mask_image.ipynb". To see the step by step implementation of the code use "Shift+Enter"._**

**For detecting the mask in live stream run "detect_mask_video.py" with command**

```
python detect_mask_video.py <path to face detection model,default = 'face_detector',optional>  <path to mask detection model,default = 'mask_detector.model',optional> <minimum probability to filter the weak detection, default = 0.5, optional>

Example: "python detect_mask_video.py"
```
