import cv2
import random
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import pyautogui
import time
import pickle as pkl

import json
import numpy as np
import pickle as pkl
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image

# Get the screen size
screen_width, screen_height = pyautogui.size()
top_left = (0, 0)
bottom_right = (screen_width - 1, screen_height - 1)

# Create an FaceLandmarker object.
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# Capture video from webcam
cap = cv2.VideoCapture(2)

def create_dataset():
  fps = 30
  prev_frame_time = 0
  new_frame_time = 0

  dataset = []

  while cap.isOpened():
      success, cv2_image = cap.read()
      if not success:
          print("Ignoring empty camera frame.")
          continue

      cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
      image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(cv2_image))
      detection_result = detector.detect(image)
      annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
      
      # Get the current mouse position
      mouse_x, mouse_y = pyautogui.position()
      print(f"Mouse position: ({mouse_x}, {mouse_y})")

      # Display the image
      annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
      height = 200
      width = int(annotated_image.shape[1] * height / annotated_image.shape[0])
      resized_image = cv2.resize(annotated_image, (width, height))

      new_frame_time = time.time()
      fps = 1 / (new_frame_time - prev_frame_time)
      prev_frame_time = new_frame_time

      cv2.putText(resized_image, f"FPS: {int(fps)}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1, cv2.LINE_AA)

      cv2.imshow('My Webcam', resized_image)
      if cv2.waitKey(int(1000/30)) & 0xFF == 27:  # Press 'ESC' to exit
          break

      # Extract face blendshapes and mouse position
      if detection_result.face_blendshapes:
          face_landmarks = detection_result.face_landmarks[0]
          left_eye_flattened = [384, 385, 386, 387, 388, 390, 263, 398, 276, 282, 283, 285, 293, 295, 296, 300, 334, 336, 466, 474, 475, 476, 477, 362, 373, 374, 249, 380, 381, 382]
          right_eye_flattened = [133, 7, 144, 145, 153, 154, 155, 157, 158, 159, 160, 33, 161, 163, 173, 46, 52, 53, 55, 63, 65, 66, 70, 469, 470, 471, 472, 105, 107, 246]
          left_eye_landmarks = [face_landmarks[i] for i in left_eye_flattened]
          right_eye_landmarks = [face_landmarks[i] for i in right_eye_flattened]

          left_eye_frame, right_eye_frame = extract_eyes(cv2_image, left_eye_landmarks, right_eye_landmarks)

          # Convert eye frames to grayscale
          left_eye_frame_gray = cv2.cvtColor(left_eye_frame, cv2.COLOR_RGB2GRAY)
          right_eye_frame_gray = cv2.cvtColor(right_eye_frame, cv2.COLOR_RGB2GRAY)

          # Stack left eye on top of right
          eye_frames = np.vstack((left_eye_frame_gray, right_eye_frame_gray))

          row = {}
          row['eye_frames'] = eye_frames
          row['mouseX'] = mouse_x
          row['mouseY'] = mouse_y
          dataset.append(row)

  cap.release()
  cv2.destroyAllWindows()
  with open('dataset.pkl', 'wb') as f:
      pkl.dump(dataset, f)

  plot_mouse_distribution(dataset)
  return None


def extract_eyes(frame, left_eye_landmarks, right_eye_landmarks):
    eye_frames = []
    
    for eye_landmarks in [left_eye_landmarks, right_eye_landmarks]:
        # Get the eye region bounding box
        min_x = int(min(eye_landmarks, key=lambda x: x.x).x * frame.shape[1])
        max_x = int(max(eye_landmarks, key=lambda x: x.x).x * frame.shape[1])
        min_y = int(min(eye_landmarks, key=lambda x: x.y).y * frame.shape[0])
        max_y = int(max(eye_landmarks, key=lambda x: x.y).y * frame.shape[0])
        
        # Crop the eye region from the frame
        eye_frame = frame[min_y:max_y, min_x:max_x]
        
        # Resize the eye frame to a fixed size
        eye_frame = cv2.resize(eye_frame, (100, 60))
        
        eye_frames.append(eye_frame)

    return eye_frames

def plot_mouse_distribution(dataset):
    mouse_x = [row['mouseX'] for row in dataset]
    mouse_y = [row['mouseY'] for row in dataset]

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(mouse_x, bins=50, color='blue', alpha=0.7)
    plt.title('Distribution of mouseX values')
    plt.xlabel('mouseX')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(mouse_y, bins=50, color='green', alpha=0.7)
    plt.title('Distribution of mouseY values')
    plt.xlabel('mouseY')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('mouse_distribution.png')
    plt.show()

class EyeTrackingModel(nn.Module):
    def __init__(self):
        super(EyeTrackingModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(64 * 25 * 30, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(-1, 64 * 30 * 25)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

def train_model():
    with open('dataset.pkl', 'rb') as f:
        dataset = pkl.load(f)
    random.shuffle(dataset)
    print('Total dataset size:', len(dataset))

    # Split the dataset into features (X) and target variables (y)
    X = []
    y = []
    for data in dataset[::1]:
        features = data['eye_frames']
        X.append(features)
        y.append([data['mouseX'], data['mouseY']])

    X = np.array(X, dtype=np.float32) / 255
    y = np.array(y, dtype=np.float32)

    # Normalize mouseX and mouseY using top-left and bottom-right coordinates
    y[:, 0] = (y[:, 0] - top_left[0]) / (bottom_right[0] - top_left[0])
    y[:, 1] = (y[:, 1] - top_left[1]) / (bottom_right[1] - top_left[1])

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert data to PyTorch tensors
    X_train = torch.from_numpy(X_train).unsqueeze(1)
    X_test = torch.from_numpy(X_test).unsqueeze(1)
    y_train = torch.from_numpy(y_train)
    y_test = torch.from_numpy(y_test)

    # Create the model, loss function, and optimizer
    model = EyeTrackingModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 10
    batch_size = 32
    for epoch in range(num_epochs):
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Evaluate the model and print results
    with torch.no_grad():
        outputs = model(X_test)
        mse = criterion(outputs, y_test)
        print(f"Mean Squared Error: {mse:.4f}")

    # Save the model
    torch.save(model.state_dict(), 'eye_tracking_model.pth')

def infer():
    pyautogui.FAILSAFE = False
    # Load the model
    model = EyeTrackingModel()
    model.load_state_dict(torch.load('eye_tracking_model.pth'))
    model.eval()

    # Start inference
    cap = cv2.VideoCapture(2)
    while cap.isOpened():
        success, cv2_image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(cv2_image))
        detection_result = detector.detect(image)

        if detection_result.face_blendshapes:
            face_landmarks = detection_result.face_landmarks[0]
            left_eye_flattened = [384, 385, 386, 387, 388, 390, 263, 398, 276, 282, 283, 285, 293, 295, 296, 300, 334, 336, 466, 474, 475, 476, 477, 362, 373, 374, 249, 380, 381, 382]
            right_eye_flattened = [133, 7, 144, 145, 153, 154, 155, 157, 158, 159, 160, 33, 161, 163, 173, 46, 52, 53, 55, 63, 65, 66, 70, 469, 470, 471, 472, 105, 107, 246]
            left_eye_landmarks = [face_landmarks[i] for i in left_eye_flattened]
            right_eye_landmarks = [face_landmarks[i] for i in right_eye_flattened]

            left_eye_frame, right_eye_frame = extract_eyes(cv2_image, left_eye_landmarks, right_eye_landmarks)

            # Convert eye frames to grayscale
            left_eye_frame_gray = cv2.cvtColor(left_eye_frame, cv2.COLOR_RGB2GRAY)
            right_eye_frame_gray = cv2.cvtColor(right_eye_frame, cv2.COLOR_RGB2GRAY)

            # Stack left eye on top of right
            eye_frames = np.vstack((left_eye_frame_gray, right_eye_frame_gray)) / 255

            # Convert eye frames to PyTorch tensor
            eye_frames_tensor = torch.from_numpy(eye_frames).float().unsqueeze(0).unsqueeze(0)

            # Get the predicted coordinates
            with torch.no_grad():
                prediction = model(eye_frames_tensor)

            mouse_x = prediction[0][0].item()
            mouse_y = prediction[0][1].item()

            # Denormalize the predicted coordinates
            mouse_x = mouse_x * (bottom_right[0] - top_left[0]) + top_left[0]
            mouse_y = mouse_y * (bottom_right[1] - top_left[1]) + top_left[1]
            print('Mouse X:', mouse_x, '| Mouse Y:', mouse_y)

            # Move the mouse to the predicted coordinates
            pyautogui.moveTo(mouse_x, mouse_y)

        # show camera window with facelandmarks
        annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        height = 200
        width = int(annotated_image.shape[1] * height / annotated_image.shape[0])
        resized_image = cv2.resize(annotated_image, (width, height))
        cv2.imshow('Face Landmarks', resized_image)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
  #create_dataset()
  #train_model()
  infer()