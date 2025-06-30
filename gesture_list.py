import cv2
import os

# Gesture names
gesture_list = ['peace', 'thumbs_up', 'heart', 'ok', 'point','fist','wave','ily']

print("Multi-HandGesture Data Collection")
print("Hand Gestures to collect:", gesture_list)
print("Here's how to do :D")
print("Open your camera and press 'c' to capture images of your hand gesture.")
print("Press 'n' to enter next gesture name, 'q' to quit.")

cap = cv2.VideoCapture(0)
gesture_index = 0
gesture_name = gesture_list[gesture_index]
save_dir = f"data/{gesture_name}"
os.makedirs(save_dir, exist_ok=True)
count = 0

print(f"Start capturing for gesture: '{gesture_name}'")

while True:
    ret, frame = cap.read()
    if not ret:
        break


    display_text = f"Gesture: {gesture_name} | Image: {count}"
    cv2.putText(frame, display_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Gesture Collector", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        # SAVE
        filepath = f"{save_dir}/{count}.jpg"
        cv2.imwrite(filepath, frame)
        print(f"Captured {filepath}")
        count += 1

    elif key == ord('n'):
        # Next
        gesture_index += 1
        if gesture_index >= len(gesture_list):
            print("All gestures collected.")
            break
        gesture_name = gesture_list[gesture_index]
        save_dir = f"data/{gesture_name}"
        os.makedirs(save_dir, exist_ok=True)
        count = 0
        print(f"➡️ Now capturing for gesture: '{gesture_name}'")

    elif key == ord('q'):
        print("Quitting capture.")
        break

cap.release()
cv2.destroyAllWindows()
