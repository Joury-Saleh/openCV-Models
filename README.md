# Computer Vision Models Setup Guide

This repository contains several computer vision applications for face detection, age/gender classification, eye blink detection, and object detection. This guide will help you set up all the necessary model files and dependencies.

## Table of Contents
- [Project Overview](#project-overview)
- [Required Model Files](#required-model-files)
- [Installation Steps](#installation-steps)
- [Download Links](#download-links)
- [Directory Structure](#directory-structure)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)

## Project Overview

The repository contains four main applications:

1. **Gender-Age-EyeBlink-Detection-vid.py** - Video-based age, gender, and blink detection
2. **Gender-Age-EyeBlinks-Detection-cam.py** - Real-time camera-based detection with face tracking
3. **Age-Gender-Detect.py** - Static image age and gender detection
4. **obj-names-model.py** - General object detection

## Required Model Files

### For Face Landmark Detection (All Blink Detection Scripts)
- **shape_predictor_68_face_landmarks.dat** (68.7 MB)
  - Used for detecting facial landmarks and eye regions
  - Required for eye blink detection

### For Age Detection
- **age_deploy.prototxt** / **deploy_age.prototxt** (1 KB)
  - Network architecture for age classification
- **age_net.caffemodel** (513 MB)
  - Pre-trained weights for age detection

### For Gender Detection
- **gender_deploy.prototxt** / **deploy_gender.prototxt** (1 KB)
  - Network architecture for gender classification
- **gender_net.caffemodel** (513 MB)
  - Pre-trained weights for gender detection

### For Face Detection (Age-Gender-Detect.py)
- **haarcascade_frontalface_alt.xml** (930 KB)
  - Haar cascade classifier for face detection

### For Object Detection
- **frozen_inference_graph.pb** (20.2 MB)
  - Pre-trained MobileNet SSD model
- **ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt** (29 KB)
  - Configuration file for MobileNet SSD
- **thing.names** (1 KB)
  - Class names file (COCO dataset classes)

## Download Links

### 1. Shape Predictor (Facial Landmarks)
```bash
# Download from dlib's official repository
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
```
**Alternative link:** [Direct Download](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

### 2. Age and Gender Models
The age and gender models are from the OpenCV Face Recognition project:

**Age Model Files:**
- [age_deploy.prototxt](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt)
- [age_net.caffemodel](https://github.com/GilLevi/AgeGenderDeepLearning/releases/download/v0.1/age_net.caffemodel)

**Gender Model Files:**
- [gender_deploy.prototxt](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt)
- [gender_net.caffemodel](https://github.com/GilLevi/AgeGenderDeepLearning/releases/download/v0.1/gender_net.caffemodel)

### 3. Face Detection (Haar Cascade)
```bash
# Download from OpenCV repository
wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt.xml
```

### 4. Object Detection Models
**MobileNet SSD:**
- [frozen_inference_graph.pb](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v3_large_coco_2020_01_14.tar.gz)
- [ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt](https://github.com/opencv/opencv_extra/blob/master/testdata/dnn/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt)

**COCO Classes:**
- [thing.names (COCO classes)](https://github.com/pjreddie/darknet/blob/master/data/coco.names)

## Directory Structure

Organize your files as follows:

```
project_root/
│
├── Gender-Age-EyeBlink-Detection-vid.py
├── Gender-Age-EyeBlinks-Detection-cam.py
├── Age-Gender-Detect.py
├── obj-names-model.py
├── shape_predictor_68_face_landmarks.dat
├── age_deploy.prototxt
├── age_net.caffemodel
├── gender_deploy.prototxt
├── gender_net.caffemodel
├── old-women.mp4                    # Your video file
├── data/
│   ├── deploy_age.prototxt
│   ├── age_net.caffemodel
│   ├── deploy_gender.prototxt
│   ├── gender_net.caffemodel
│   └── haarcascade_frontalface_alt.xml
├── files/
│   ├── frozen_inference_graph.pb
│   ├── ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt
│   └── thing.names
└── images/
    ├── girl1.jpg                    # Your test image
    └── employee.png                 # Your test image
```

## Installation Steps

### Step 1: Install Python Dependencies
```bash
pip install opencv-python
pip install dlib
pip install imutils
pip install scipy
pip install numpy
```

### Step 2: Download Model Files

#### Option A: Automated Download Script
Create a download script to get all models:

```bash
#!/bin/bash
# Create directories
mkdir -p data files images

# Download shape predictor
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

# Download age/gender models
cd data
wget https://github.com/GilLevi/AgeGenderDeepLearning/releases/download/v0.1/age_net.caffemodel
wget https://github.com/GilLevi/AgeGenderDeepLearning/releases/download/v0.1/gender_net.caffemodel
wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt.xml

# Download object detection models
cd ../files
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v3_large_coco_2020_01_14.tar.gz
tar -xzf ssd_mobilenet_v3_large_coco_2020_01_14.tar.gz
cp ssd_mobilenet_v3_large_coco_2020_01_14/frozen_inference_graph.pb .
wget https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names -O thing.names

cd ..
```

#### Option B: Manual Download
1. Visit each link above and download the files manually
2. Place them in the appropriate directories as shown in the structure

### Step 3: Create Prototxt Files

Create the missing prototxt files for age/gender detection:

**age_deploy.prototxt:**
```prototxt
name: "AgeNet"
input: "data"
input_dim: 1
input_dim: 3
input_dim: 227
input_dim: 227
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 96
    kernel_size: 7
    stride: 4
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
# ... (complete prototxt content available in OpenCV samples)
```

## Dependencies

### Python Packages
```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
opencv-python>=4.5.0
dlib>=19.21.0
imutils>=0.5.4
scipy>=1.6.0
numpy>=1.19.0
```

### System Requirements
- Python 3.7+
- OpenCV 4.x
- dlib (requires CMake and Visual Studio on Windows)
- Webcam (for real-time detection scripts)

### Installing dlib (if pip fails)
**On Ubuntu/Debian:**
```bash
sudo apt-get install build-essential cmake
sudo apt-get install libopenblas-dev liblapack-dev
sudo apt-get install libx11-dev libgtk-3-dev
pip install dlib
```

**On Windows:**
```bash
# Install Visual Studio Build Tools first
pip install cmake
pip install dlib
```

**On macOS:**
```bash
brew install cmake
pip install dlib
```

## Usage

### 1. Video-based Detection
```bash
python Gender-Age-EyeBlink-Detection-vid.py
```
- Processes a video file (`old-women.mp4`)
- Detects age, gender, and eye blinks
- Press 'q' to quit

### 2. Real-time Camera Detection
```bash
python Gender-Age-EyeBlinks-Detection-cam.py
```
- Uses webcam for real-time detection
- Advanced face tracking with persistent counters
- Controls: 'r' (reset), 'c' (clear), 's' (stats), 'q' (quit)

### 3. Image-based Age/Gender Detection
```bash
python Age-Gender-Detect.py
```
- Processes static images
- Modify image path in script
- Press 's' to save results

### 4. Object Detection
```bash
python obj-names-model.py
```
- Detects common objects in images
- Uses COCO dataset classes
- Modify image path as needed

## Troubleshooting

### Common Issues

**1. "FileNotFoundError: shape_predictor_68_face_landmarks.dat"**
- Download the file using the provided link
- Ensure it's in the root directory

**2. "Error loading age/gender models"**
- Check if .caffemodel files are downloaded correctly
- Verify prototxt files exist and have correct format

**3. "ImportError: No module named 'dlib'"**
- Install build tools first, then dlib
- On Windows, ensure Visual Studio Build Tools are installed

**4. Low detection accuracy**
- Ensure good lighting conditions
- Check camera resolution and positioning
- Verify model files are not corrupted

**5. "Cannot open video file"**
- Check video file path and format
- Ensure OpenCV supports the video codec

### Performance Optimization

- **Reduce video resolution** for faster processing
- **Adjust detection thresholds** based on your needs
- **Use GPU acceleration** if available (requires OpenCV with CUDA)

### Model File Verification

Check if files downloaded correctly:
```bash
# Check file sizes (approximate)
ls -lh shape_predictor_68_face_landmarks.dat  # ~68MB
ls -lh data/age_net.caffemodel                # ~513MB
ls -lh data/gender_net.caffemodel             # ~513MB
ls -lh files/frozen_inference_graph.pb        # ~20MB
```

## License

These models are subject to their respective licenses:
- **dlib models**: Boost Software License
- **OpenCV models**: BSD License
- **Age/Gender models**: Academic use (check original papers)
- **MobileNet**: Apache 2.0 License



---

