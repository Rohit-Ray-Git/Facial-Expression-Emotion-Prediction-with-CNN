# Facial Expression Mood Prediction with CNN 😊😞

## Overview
This project uses Convolutional Neural Networks (CNNs) to predict moods based on facial expressions. The model is trained on a dataset of facial images labeled with corresponding mood states (e.g., Happy, Sad). The application allows users to upload images and receive mood predictions based on facial features detected in the image.

## Features 🚀
- **Mood Prediction**: Predicts whether the mood in an image is "Happy" or "Sad".
- **CNN Model**: Utilizes Convolutional Neural Networks to perform facial emotion recognition.
- **Real-time Prediction**: Upload an image and get instant mood predictions.
- **Streamlit Interface**: A simple, user-friendly interface built with Streamlit for easy image uploads and predictions.

## Technologies Used 💻
- **Python** 🐍
- **TensorFlow/Keras**: For building and training the CNN model.
- **Streamlit**: For creating the web interface.
- **OpenCV**: For image preprocessing and manipulation.
- **NumPy**: For array manipulations.
- **Matplotlib**: For visualizations.

## Model Description 🧠
The model is a Convolutional Neural Network (CNN) that takes in images of faces and outputs a binary prediction of the mood: either "Happy" or "Sad". The architecture includes:

- 3 convolutional layers for feature extraction.
- Max-pooling layers to reduce dimensionality.
- A fully connected layer to make the final mood prediction.

The model was trained on a dataset of facial images, with each image labeled as either "Happy" or "Sad". The training process used an RMSprop optimizer, binary cross-entropy loss function, and accuracy as the primary metric.

## How to Use 🛠️

1. **Start the Streamlit App**:
   After the dependencies are installed, start the app with the following command:

   ```
   streamlit run app.py
   ```

   
2. **Upload an Image**:

- The app allows you to upload an image of a face (in .jpg, .png, .jpeg formats).
- The model will predict whether the mood is "Happy" 😊 or "Sad" 😞 based on the facial expression in the image.

3. **See the Results** ✅: After uploading the image, the app will display the predicted mood along with the actual label if available.

## Note 📌
While the model performs well on the available dataset, there may be occasional misclassifications in certain cases, especially if the image quality is poor, the face is obscured, or there are variations in lighting or pose. Additionally, the model is currently trained to predict only two moods (Happy 😊 and Sad 😞), and expanding to more mood categories may require more diverse data and further training.

It's important to keep in mind that emotion recognition from facial expressions can be subjective and influenced by various factors. The model's accuracy may improve with more training data, better preprocessing, and fine-tuning.

