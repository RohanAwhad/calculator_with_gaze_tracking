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
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

# Print the screen coordinates
print(f"Top-left coordinate: {top_left}")
print(f"Bottom-right coordinate: {bottom_right}")

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
          row = {}
          face_landmarks = detection_result.face_landmarks[0]
          for i, landmark in enumerate(face_landmarks):
            row[f'landmark_{i}_x'] = landmark.x
            row[f'landmark_{i}_y'] = landmark.y
            row[f'landmark_{i}_z'] = landmark.z

          face_blendshapes = detection_result.face_blendshapes[0]
          for category in face_blendshapes:
              if category.category_name in ['browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight', 'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight']:
                  row[category.category_name] = category.score
          row['mouseX'] = mouse_x
          row['mouseY'] = mouse_y
          dataset.append(row)

  cap.release()
  cv2.destroyAllWindows()
  with open('dataset.pkl', 'wb') as f:
      pkl.dump(dataset, f)

  plot_mouse_distribution(dataset)

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


LANDMARK_FEATURES = [f'landmark_{i}_{a}' for i in range(0, 478) for a in 'xyz']
BLENDSHAPE_FEATURES = ['browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight', 'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight']
FEATURES = LANDMARK_FEATURES + BLENDSHAPE_FEATURES

class LL(nn.Module):
  def __init__(self, in_dim, out_dim, dropout_rate):
    super().__init__()
    self.hidden = nn.Linear(in_dim, out_dim)
    self.dropout = nn.Dropout(dropout_rate)
    self.layernorm = nn.LayerNorm(out_dim)
    self.act = nn.ReLU()

  def forward(self, x):
    x = self.hidden(x)
    x = self.dropout(x)
    x = self.layernorm(x)
    x = self.act(x)
    return x

class FaceLandmarkRegressor(nn.Module):
    def __init__(self, num_features, num_hidden=32, dropout_rate=0.2):
        super(FaceLandmarkRegressor, self).__init__()
        self.layers = nn.Sequential(
          LL(num_features, num_hidden, dropout_rate),
          LL(num_hidden, num_hidden, dropout_rate),
          LL(num_hidden, num_hidden, dropout_rate),
        )
        self.output = nn.Linear(num_hidden, 2)

    def forward(self, x):
        x = self.layers(x)
        x = self.output(x)
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
        features = [data[feature] for feature in FEATURES]
        X.append(features)
        y.append([data['mouseX'], data['mouseY']])

    X = np.array(X)
    y = np.array(y)

    # Normalize mouseX and mouseY using top-left and bottom-right coordinates
    y[:, 0] = (y[:, 0] - top_left[0]) / (bottom_right[0] - top_left[0])
    y[:, 1] = (y[:, 1] - top_left[1]) / (bottom_right[1] - top_left[1])

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features using StandardScaler
    feat_scaler = StandardScaler()
    X_train = feat_scaler.fit_transform(X_train)
    X_test = feat_scaler.transform(X_test)

    # Scale the labels using StandardScaler
    label_scaler = StandardScaler()
    #y_train = label_scaler.fit_transform(y_train)
    #y_test = label_scaler.transform(y_test)

    # Train the PyTorch regression model
    model = FaceLandmarkRegressor(num_features=len(FEATURES))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    batch_size = 128
    num_epochs = 100
    loss_history = []
    model.train()

    # Convert X_train and y_train to PyTorch tensors
    inputs = torch.from_numpy(X_train).float()
    targets = torch.from_numpy(y_train).float()

    # Create a TensorDataset from inputs and targets
    train_dataset = TensorDataset(inputs, targets)

    # Create a DataLoader with batch size 32
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    for epoch in range(num_epochs):
        for batch_inputs, batch_targets in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        loss_history.append(loss.item())


    # Evaluate the model and print results
    model.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(X_test).float()
        targets = torch.from_numpy(y_test).float()
        outputs = model(inputs)
        #outputs = label_scaler.inverse_transform(outputs.numpy())
        #targets = label_scaler.inverse_transform(targets.numpy())

        mse = torch.mean((outputs - targets) ** 2)
        print(f"Mean Squared Error: {mse:.4f}")

    # Save the model and scaler
    torch.save(model.state_dict(), 'torch_model.pth')
    with open('scaler.pkl', 'wb') as f:
        pkl.dump((feat_scaler, label_scaler), f)

    # Plot loss curve and save it as PNG
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.savefig('loss_curve.png')


def infer():
    pyautogui.FAILSAFE = False
    # Load the model, scaler, and config
    model = FaceLandmarkRegressor(num_features=len(FEATURES))
    model.load_state_dict(torch.load('torch_model.pth'))
    model.eval()
    with open('scaler.pkl', 'rb') as f:
        feat_scaler, label_scaler = pkl.load(f)
    with open('config.json', 'r') as f:
        config = json.load(f)

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
            features = {}
            face_landmarks = detection_result.face_landmarks[0]
            for i, landmark in enumerate(face_landmarks):
              features[f'landmark_{i}_x'] = landmark.x
              features[f'landmark_{i}_y'] = landmark.y
              features[f'landmark_{i}_z'] = landmark.z

            face_blendshapes = detection_result.face_blendshapes[0]
            features.update({category.category_name: category.score for category in face_blendshapes if category.category_name in FEATURES})
            features = [features[k] for k in FEATURES]
            features = np.array(features).reshape(1, -1)
            features = feat_scaler.transform(features)

            with torch.no_grad():
                inputs = torch.from_numpy(features).float()
                prediction = model(inputs)
                #prediction = label_scaler.inverse_transform(prediction.numpy())

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
  train_model()
  infer()
