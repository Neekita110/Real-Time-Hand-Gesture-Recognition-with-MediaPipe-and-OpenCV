import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load the model
model_path = r'C:\Users\hp\Hand..gasture\my_model.h5'
model = load_model(model_path)

# Load the label encoder (manually specify labels used during training)
labels = ['All_okh', 'Not_okh', 'Help', 'Stop']  # Replace with your actual gesture labels
label_encoder = LabelEncoder()
label_encoder.fit(labels)

# Initialize webcam
cap = cv2.VideoCapture(0)

def preprocess_image(image):
    image = cv2.resize(image, (100, 100))  # Resize to the size used during training
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    processed_frame = preprocess_image(frame)

    # Predict the gesture
    predictions = model.predict(processed_frame)
    predicted_label = label_encoder.inverse_transform([np.argmax(predictions)])

    # Display the prediction on the frame
    cv2.putText(frame, f'Gesture: {predicted_label[0]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
