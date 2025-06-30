import cv2
import numpy as np
import tensorflow as tf
import os


model = tf.keras.models.load_model("")

img_height, img_width = 128, 128  

data_dir = "data"
class_names = sorted(os.listdir(data_dir))
print("Loaded gesture classes:", class_names)


cap = cv2.VideoCapture(0)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    x1, y1 = 100, 100
    x2, y2 = x1 + 200, y1 + 200
    roi = frame[y1:y2, x1:x2]

    # Preprocess the region of interest
    img = cv2.resize(roi, (img_width, img_height))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Make prediction
    predictions = model.predict(img)
    predicted_index = np.argmax(predictions[0])
    predicted_label = class_names[predicted_index]
    confidence = predictions[0][predicted_index]

    # Show prediction
    label = f"{predicted_label} ({confidence*100:.1f}%)"
    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()