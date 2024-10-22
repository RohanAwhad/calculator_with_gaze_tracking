## Video Demo

https://github.com/user-attachments/assets/51c73857-4b3d-4112-b51d-2dab16bcf766



### Notes:

face_landmarks_list is a list of all the faces, which in turn have the landmarks

class Blendshapes(enum.IntEnum):
  """The 52 blendshape coefficients."""

  BROW_DOWN_LEFT = 1
  BROW_DOWN_RIGHT = 2
  BROW_INNER_UP = 3
  BROW_OUTER_UP_LEFT = 4
  BROW_OUTER_UP_RIGHT = 5
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
