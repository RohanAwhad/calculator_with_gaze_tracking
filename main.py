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



landmark_ids = set([263, 249, 249, 390, 390, 373, 373, 374, 374, 380, 380, 381, 381, 382, 382, 362, 263, 466, 466, 388, 388, 387, 387, 386, 386, 385, 385, 384, 384, 398, 398, 362, 276, 283, 283, 282, 282, 295, 295, 285, 300, 293, 293, 334, 334, 296, 296, 336, 474, 475, 475, 476, 476, 477, 477, 474, 33, 7, 7, 163, 163, 144, 144, 145, 145, 153, 153, 154, 154, 155, 155, 133, 33, 246, 246, 161, 161, 160, 160, 159, 159, 158, 158, 157, 157, 173, 173, 133, 46, 53, 53, 52, 52, 65, 65, 55, 70, 63, 63, 105, 105, 66, 66, 107, 469, 470, 470, 471, 471, 472, 472, 469])
#landmark_ids = range(478)

LANDMARK_FEATURES = [f'landmark_{i}_{a}' for i in landmark_ids for a in 'xyz']
BLENDSHAPE_FEATURES = ['browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight', 'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight']
FEATURES = LANDMARK_FEATURES + BLENDSHAPE_FEATURES

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

    # Train the Ridge regression model
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    # Evaluate the model and print results
    y_pred = model.predict(X_test)
    mse = np.mean((y_pred - y_test) ** 2)
    print(f"Mean Squared Error: {mse:.4f}")

    # Save the model and scaler
    with open('ridge_model.pkl', 'wb') as f:
        pkl.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pkl.dump(feat_scaler, f)


def infer():
    pyautogui.FAILSAFE = False
    # Load the model, scaler, and config
    with open('ridge_model.pkl', 'rb') as f:
        model = pkl.load(f)
    with open('scaler.pkl', 'rb') as f:
        feat_scaler = pkl.load(f)
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

            prediction = model.predict(features)

            mouse_x = prediction[0][0]
            mouse_y = prediction[0][1]

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


'''
if __name__ == '__main__':
  create_dataset()
  train_model()
  infer()

'''

# write me a function that uses the media pipe to get the landmarks, and uses those landmarks to crop out left eye and right eye from the frame and save it on disk, for just a single frame
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
        
        eye_frames.append(eye_frame)

    return eye_frames

def save_eyes_to_disk(left_eye_frame, right_eye_frame):
    cv2.imwrite("left_eye.jpg", cv2.cvtColor(left_eye_frame, cv2.COLOR_RGB2BGR))
    cv2.imwrite("right_eye.jpg", cv2.cvtColor(right_eye_frame, cv2.COLOR_RGB2BGR))

def extract_eyes_and_save_to_disk(frame, detection_result):
    face_landmarks = detection_result.face_landmarks[0]

    left_eye_flattened = [384, 385, 386, 387, 388, 390, 263, 398, 276, 282, 283, 285, 293, 295, 296, 300, 334, 336, 466, 474, 475, 476, 477, 362, 373, 374, 249, 380, 381, 382]
    right_eye_flattened = [133, 7, 144, 145, 153, 154, 155, 157, 158, 159, 160, 33, 161, 163, 173, 46, 52, 53, 55, 63, 65, 66, 70, 469, 470, 471, 472, 105, 107, 246]
    left_eye_landmarks = [face_landmarks[i] for i in left_eye_flattened]
    right_eye_landmarks = [face_landmarks[i] for i in right_eye_flattened]

    #left_eye_landmarks = [face_landmarks[i] for i in range(33, 133)]
    #right_eye_landmarks = [face_landmarks[i] for i in range(133, 233)]

    left_eye_frame, right_eye_frame = extract_eyes(frame, left_eye_landmarks, right_eye_landmarks)

    save_eyes_to_disk(left_eye_frame, right_eye_frame)


# Let the first few frames be warmup and discard them
for _ in range(5):
    success, cv2_image = cap.read()

success, cv2_image = cap.read()
cv2.imwrite('frame.jpg', cv2_image)
cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(cv2_image))
detection_result = detector.detect(image)
extract_eyes_and_save_to_disk(cv2_image, detection_result)
