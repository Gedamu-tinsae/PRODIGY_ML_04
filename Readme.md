# Hand Gesture Recognition Model

This project uses a Convolutional Neural Network (CNN) to recognize hand gestures from images captured by a Leap Motion sensor. The model enables intuitive human-computer interaction and gesture-based control systems.

### Downloading the Dataset

Since GitHub does not allow large file uploads, you need to download the dataset from Kaggle:

1. Visit the [LeapGestRecog dataset page on Kaggle](https://www.kaggle.com/datasets/gti-upm/leapgestrecog).
2. Log in to your Kaggle account.
3. Click on the "Download" button to download the dataset.
4. Extract the downloaded archive and place the contents in a directory named `Archive` in the root of this project.


## Project Structure

```
.
├── Archive
│   ├── 00
│   │   ├── 01_palm
│   │   │   ├── frame_197957_r.png
│   │   │   ├── ...
│   │   ├── 02_l
│   │   │   ├── frame_198136_l.png
│   │   │   ├── ...
│   ├── ...
├── hand_gesture_recognition.ipynb
└── README.md
```

- `Archive/`: Dataset of hand gesture images.
- `hand_gesture_recognition.ipynb`: Jupyter Notebook with code for data loading, model training, and real-time gesture recognition.
- `README.md`: This file.

## Dataset

The dataset includes 10 hand gestures performed by 10 subjects (5 men and 5 women):
- Palm
- L
- Fist
- Fist Moved
- Thumb
- Index
- OK
- Palm Moved
- C
- Down

## Requirements

- Python 3.6+
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

Install dependencies:

pip install tensorflow opencv-python numpy matplotlib scikit-learn

## Model Training

1. Load and preprocess the dataset: Resize, normalize, and split images into training, validation, and test sets.
2. Build the CNN model: Define a sequential model with convolutional, pooling, and dense layers.
3. Train the model: Train on the training set, validate on the validation set.
4. Evaluate the model: Test the model on the test set.

## Real-Time Gesture Recognition

Use the webcam to capture and classify gestures in real-time:

1. Capture frames from the webcam.
2. Preprocess each frame: Convert to grayscale, resize, and normalize.
3. Predict the gesture: Use the trained model to classify the gesture.
4. Display the result: Show the predicted gesture on the frame.

## Running the Notebook

1. Clone the repository:

    git clone <repository-url>
    cd <repository-directory>

2. Open the Jupyter Notebook:

    jupyter notebook hand_gesture_recognition.ipynb

3. Follow the notebook instructions to train the model and perform real-time gesture recognition.
