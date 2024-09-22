import cv2
import mediapipe as mp

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt

'''
face_landmarks_list is a list of all the faces, which in turn have the landmarks

class Blendshapes(enum.IntEnum):
  """The 52 blendshape coefficients."""

  NEUTRAL = 0
  BROW_DOWN_LEFT = 1
  BROW_DOWN_RIGHT = 2
  BROW_INNER_UP = 3
  BROW_OUTER_UP_LEFT = 4
  BROW_OUTER_UP_RIGHT = 5
  CHEEK_PUFF = 6
  CHEEK_SQUINT_LEFT = 7
  CHEEK_SQUINT_RIGHT = 8
  EYE_BLINK_LEFT = 9
  EYE_BLINK_RIGHT = 10
  EYE_LOOK_DOWN_LEFT = 11
  EYE_LOOK_DOWN_RIGHT = 12
  EYE_LOOK_IN_LEFT = 13
  EYE_LOOK_IN_RIGHT = 14
  EYE_LOOK_OUT_LEFT = 15
  EYE_LOOK_OUT_RIGHT = 16
  EYE_LOOK_UP_LEFT = 17
  EYE_LOOK_UP_RIGHT = 18
  EYE_SQUINT_LEFT = 19
  EYE_SQUINT_RIGHT = 20
  EYE_WIDE_LEFT = 21
  EYE_WIDE_RIGHT = 22


Getting Blendshape results: detection_result.face_blendshapes -> list[list[Category(index=0, score=2.0494850616614713e-07, display_name='', category_name='_neutral')]
- category_name will vary, and is in camelcase but like above. e.g.: eyeBlinkRight
- What does the score, represent?
	- Definition: Face blendshapes are numerical values that represent the intensity of various facial expressions or muscle movements.
	- Purpose: They are used to describe and quantify facial expressions, allowing for detailed analysis or recreation of facial movements.
	- Output: The MediaPipe Face Landmarker task can output 52 different blendshape scores for each detected face.
	- Range: Each blendshape score is typically a float value between 0.0 and 1.0, where 0.0 means the expression is not present, and 1.0 means it's fully expressed.
'''

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

# STEP 1: Import the necessary modules.
# TODO: move the imports above

# STEP 2: Create an FaceLandmarker object.
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize drawing utility
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
iris_drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)

# Capture video from webcam
cap = cv2.VideoCapture(2)

while cap.isOpened():
    success, cv2_image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(cv2_image))
    detection_result = detector.detect(image)
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    
    # Display the image
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    height = 200
    width = int(annotated_image.shape[1] * height / annotated_image.shape[0])
    resized_image = cv2.resize(annotated_image, (width, height))

    cv2.imshow('My Webcam', resized_image)
    if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()
