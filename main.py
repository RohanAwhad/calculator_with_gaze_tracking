import cv2

# Capture video from webcam
cap = cv2.VideoCapture(2)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Display the image
    cv2.imshow('My Webcam', image)
    if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()


'''
'''
