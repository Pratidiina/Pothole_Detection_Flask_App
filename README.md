# Road Condition Detection

## Overview

This project is a **Road Condition Detection System** that classifies images as either a **Plain Road** or a **Pothole**. It uses a Convolutional Neural Network (CNN) trained on a dataset of road images. The model is integrated with a Flask web application, allowing users to upload an image via a web interface and receive predictions, along with a display of the uploaded image.

## Features

- Trainable CNN model using TensorFlow/Keras.
- Flask-based web application for easy interaction.
- Bootstrap for a responsive and user-friendly front-end.
- Upload functionality to process and classify road images.
- Displays the uploaded image alongside the prediction result.

## Dataset

- **Plain Road Images**: 727
- **Pothole Images**: 3,395

Data preprocessing includes resizing all images to 128x128 pixels and normalizing pixel values to the range [0, 1]. Data augmentation techniques were applied to address dataset imbalance.

## Technology Stack

- **Backend**: Flask (Python)
- **Frontend**: Bootstrap (HTML, CSS)
- **Model Training**: TensorFlow/Keras
- **Image Preprocessing**: Pillow, NumPy
- **Environment**: Google Colab for model training, VS Code for local development

## Requirements

Install the following dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## How to Run

### Step 1: Clone the Repository

```bash
git clone https://github.com/ubaidalishah/Pothole_Detection_Flask_App.git
cd Pothole_Detection_Flask_App
```

### Step 2: Save the Model Locally

1. Train the model on Google Colab (or use a pre-trained model provided in this repository).
2. Save the model in HDF5 format (`model.h5`) and place it in the root directory of the project.

### Step 3: Run the Flask Application

Start the Flask server:

```bash
python app.py
```

Access the web app at `http://127.0.0.1:5000`.

### Step 4: Use the Web Interface

1. Upload an image of a road.
2. View the prediction result (Plain Road or Pothole) along with the uploaded image.

## File Structure

```
road-condition-detection/
├── app.py              # Flask application
├── model.h5            # Trained model (HDF5 format)
├── requirements.txt    # Required Python libraries
├── static/             # Static files (e.g., uploaded images)
├── templates/          # HTML templates (index.html, result.html)
└── README.md           # Project documentation
```

## Model Details

- **Architecture**: Sequential CNN
- **Layers**:
  - 3 Convolutional layers with ReLU activation and MaxPooling
  - 3 Dropout layers to prevent overfitting
  - Flatten and Dense layers for classification
- **Output**: 2 neurons with Sigmoid activation for binary classification

## Troubleshooting

- **Pillow Import Error**: Install Pillow with `pip install pillow`.
- **Model Predictions Always Show "Plain Road"**:
  - Ensure consistent preprocessing of images.
  - Check dataset balance or use class weights.
- **Static Image Not Displaying**:
  - Ensure the `static/uploads` folder exists.

## Future Improvements

- Expand the dataset for better generalization.
- Use Transfer Learning for improved accuracy.
- Add real-time video stream processing.

## License

This project is licensed under the [MIT License](LICENSE).

## Author

Ubaid Ali Shah

