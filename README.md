# AI Image Classifier

A Streamlit-based web application that uses deep learning to classify images using the MobileNetV2 model pre-trained on ImageNet dataset.

## Features

- **Easy Image Upload**: Support for JPG, JPEG, and PNG formats
- **Pre-trained Model**: Utilizes MobileNetV2 trained on ImageNet with 1000+ categories
- **Fast Classification**: Optimized model with caching for quick predictions
- **Top 3 Predictions**: Shows the three most likely classifications with confidence scores
- **Clean Interface**: Simple, intuitive Streamlit interface

## Prerequisites

- Python 3.8 or higher
- TensorFlow 2.x

## Installation

1. **Clone the repository**
```bash
git clone <your-repository-url>
cd ai-image-classifier
```

2. **Install required packages**
```bash
pip install streamlit tensorflow opencv-python numpy pillow
```

## Usage

1. **Run the application**
```bash
streamlit run app.py
```

2. **Access the web interface**
   - The application will open in your default browser
   - Default URL: `http://localhost:8501`

3. **Classify an image**
   - Click "Browse files" to upload an image (JPG, JPEG, or PNG)
   - Preview your uploaded image
   - Click "Classify Image" to get predictions
   - View the top 3 predictions with confidence scores

## How It Works

1. **Model Loading**: The app uses MobileNetV2, a lightweight convolutional neural network pre-trained on ImageNet
2. **Image Preprocessing**: Uploaded images are resized to 224x224 pixels and normalized
3. **Prediction**: The model analyzes the image and returns probability scores for 1000+ categories
4. **Results**: The top 3 most likely classifications are displayed with confidence percentages

## Project Structure
```
ai-image-classifier/
├── app.py
└── README.md
```

## Dependencies

- `streamlit`: Web application framework
- `tensorflow`: Deep learning framework
- `opencv-python`: Image processing
- `numpy`: Numerical operations
- `pillow`: Image file handling

## Technical Details

- **Model**: MobileNetV2
- **Input Size**: 224x224 pixels
- **Training Dataset**: ImageNet (1000 classes)
- **Framework**: TensorFlow/Keras
- **Preprocessing**: MobileNetV2 standard preprocessing