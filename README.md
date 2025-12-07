# FACE RECOGNITION SYSTEM AND TRACKER – “Every AI is not ML”

This program has two parts:
1. Face Recognition and Tracker using HaarCascade (LBPH)
2. Face Recognition only using MediaPipe

## FACE RECOGNITION AND TRACKER USING HAARCASCADE (LBPH)

### Installation

```bash
# Create virtual environment
python -m venv .venv

# Windows
./.venv/Scripts/activate 

# Linux/Mac
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Usage
```bash
python 01_create_dataset.py
python 02_review_dataset.py
python 03_train_model.py
python 04_predict.py
python 05_track_face.py
```

### Explanation

#### 01_create_dataset.py
Take images of faces and train the model as datasets. Make sure you are not in motion while taking images and there is no light changes. Too much light or dark changes can cause the model to be unstable. Take as many pictures as you can (Minimum 200 images).

#### 02_review_dataset.py
Review the dataset and remove any images that are not clear or not of good quality. Make sure to review all the captured images.

#### 03_train_model.py
Train the model using the dataset.

#### 04_predict.py
Predict the face using the model. 

#### 05_track_face.py
Track the face using the model.


## FACE RECOGNITION USING MEDIAPIPE

### Installation

In order to use python version 3.10 without downgrading your current python version, you use uv.

You can install it on windows with 

```bash
winget install --id AstralSoftware.UV -e
```
On mac or Linux, the right command is not yet provided.

And install the python version 3.10

```bash
winget install --id Python.Python.3.10 -e
```

Then follow the rest as normal

```bash
# Create virtual environment
uv venv --python 3.10 .venv310

# Upgrade pip
uv pip install --upgrade pip

# Windows
./.venv310/Scripts/activate 

# Linux/Mac
source .venv310/bin/activate

# Install dependencies
pip install -r requirements_mp.txt
```

### Usage
```bash
python 01_create_dataset_mp.py
python 02_review_dataset_mp.py
python 03_train_model_mp.py
python 04_predict_mp.py
```

### Explanation

#### 01_create_dataset_mp.py
Take images of faces and train the model as datasets. Make sure you are not in motion while taking images and there is no light changes. Too much light or dark changes can cause the model to be unstable. Take as many pictures as you can (Minimum 200 images).

#### 02_review_dataset_mp.py
Review the dataset and remove any images that are not clear or not of good quality. Make sure to review all the captured images.

#### 03_train_model_mp.py
Train the model using the dataset.

#### 04_predict_mp.py
Predict the face using the model. 



## License
MIT