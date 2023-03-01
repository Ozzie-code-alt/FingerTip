import mediapipe as mp
import cv2
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Set up the hand tracking
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

#  landmarks representing the fingertips
fingertip_indexes = [8, 12, 16, 20]

# Define a function to move the mouse to a target position
def move_mouse(target_position):
    screen_width, screen_height = pyautogui.size()
    x, y = target_position[0] / screen_width, target_position[1] / screen_height
    x, y = x * screen_width, y * screen_height
    pyautogui.moveTo(x, y)


# Define a function to process the output of the hand tracking solution
def process_hands(image):
    # Convert the image to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run the hand tracking solution on the image
    results = hands.process(image)

    # If hands were detected
    if results.multi_hand_landmarks:
        # For each detected hand
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the coordinates of  index
            index_finger = hand_landmarks.landmark[8]
            x, y = int(index_finger.x * image.shape[1]), int(index_finger.y * image.shape[0])
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

            # Move the mouse to the position of the index finger
            target_position = (x, y)
            move_mouse(target_position)

            # Check if the index finger is raised and click the mouse
            if index_finger.y < hand_landmarks.landmark[7].y and index_finger.y < hand_landmarks.landmark[6].y:
                pyautogui.click()

    # Convert the image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image

# you guys know this already
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = process_hands(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

