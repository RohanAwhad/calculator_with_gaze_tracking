import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from collections import deque

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
screen_width, screen_height = pyautogui.size()

def map_to_screen(x: float, y: float) -> tuple[int, int]: return (int(x * screen_width), int(y * screen_height))

def detect_index_thumb_touch(landmarks: list) -> bool:
    thumb_tip = landmarks[4]
    index_tip = landmarks[12]
    distance = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2 + (thumb_tip.z - index_tip.z)**2)**0.5
    return distance < 0.05

def exponential_moving_average(values: deque, alpha: float) -> tuple[float, float]:
    x_ema = values[0][0]
    y_ema = values[0][1]
    for i in range(1, len(values)):
        x_ema = alpha * values[i][0] + (1 - alpha) * x_ema
        y_ema = alpha * values[i][1] + (1 - alpha) * y_ema
    return x_ema, y_ema



# define a grab gesture for me.
# the function should return a bool is hand in shape of grabbed or not
def is_hand_grabbed(landmarks: list) -> bool:
    # Define the landmarks for fingertips and their corresponding base joints
    fingertips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
    finger_bases = [5, 9, 13, 17]  # Corresponding base joints

    # Check if fingertips are closer to the palm than their base joints
    is_grabbed = True
    for tip, base in zip(fingertips, finger_bases):
        if landmarks[tip].y < landmarks[base].y:
            is_grabbed = False
            break

    # Additional check for thumb
    thumb_tip = landmarks[4]
    thumb_base = landmarks[2]
    if thumb_tip.x > thumb_base.x:
        is_grabbed = False

    print('is hand in grabbing pos?', is_grabbed)
    return is_grabbed

def main():
    cap = cv2.VideoCapture(2)
    pyautogui.FAILSAFE = False
  
    position_history = deque(maxlen=10)
    alpha = 0.3
    mouse_down = False
    scroll_mode = False
    last_y = None

    frame_ctr = 0
    while cap.isOpened():
        try:
            success, image = cap.read()
            if not success:
                continue
            frame_ctr += 1

            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    index_finger_tip = hand_landmarks.landmark[8]
                    mouse_x, mouse_y = map_to_screen(index_finger_tip.x, index_finger_tip.y)
                    position_history.append((mouse_x, mouse_y))
                    if len(position_history) == 10:
                        smooth_x, smooth_y = exponential_moving_average(position_history, alpha)
                        pyautogui.moveTo(int(smooth_x), int(smooth_y))


                    # scroll gesture
                    if is_hand_grabbed(hand_landmarks.landmark):
                        if not scroll_mode:
                            scroll_mode = True
                            last_y = mouse_y
                        else:
                            scroll_amount = (mouse_y - last_y) // 10
                            pyautogui.scroll(-scroll_amount)
                            last_y = mouse_y
                    else:
                        scroll_mode = False
                        last_y = None


                    # left mouse button click gesture
                    if not scroll_mode and detect_index_thumb_touch(hand_landmarks.landmark):
                        if not mouse_down:
                            pyautogui.mouseDown(button='left')
                            mouse_down = True
                    else:
                        if mouse_down:
                            pyautogui.mouseUp(button='left')
                            mouse_down = False

            if cv2.waitKey(5) & 0xFF == 27:
                break
        except KeyboardInterrupt:
            break

    if mouse_down:
        pyautogui.mouseUp()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
