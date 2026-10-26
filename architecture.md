# Technical Design

## Data flow
1. User uploads a photo of one waste item.
2. The app resizes the image to 224 × 224 pixels.
3. A trained MobileNetV2-based classifier predicts one of two labels: `organic` or `recyclable`.
4. The app displays the predicted label and confidence.
5. If confidence is low, the user is advised to sort the item manually.

## Components
- **User interface:** Streamlit web app
- **Image processing:** Pillow and NumPy
- **Classifier:** TensorFlow/Keras transfer-learning model based on MobileNetV2
- **Training data:** User-provided labelled image folders
- **Physical hardware:** Not implemented in this software version

## Important limitations
- Only two categories are included in the starter model.
- Model quality depends on the training data and evaluation.
- Items that are dirty, mixed, occluded, or outside the training categories may be misclassified.
- The current code does not control a motor, sensor, or physical bin.
- Before adding hardware, include a safe mechanism, manual override, jam detection, and repeated testing.
