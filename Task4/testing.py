import cv2
import mediapipe as mp

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

# Function to process a single image and detect "Hi" gesture
def process_single_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image {image_path}")
        return

    # Resize the image to standard dimensions
    image = cv2.resize(image, (640, 480))

    # Convert the image to RGB for Mediapipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image with Mediapipe
    result = hands.process(rgb_image)

    # Check for hand landmarks
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks and connections on the image
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Check if the hand gesture is "Hi"
            hi_status = detect_hi_gesture(hand_landmarks)
            print(f"Hand detected: {hi_status}")

            # Overlay gesture status text on the image
            cv2.putText(image, f"Gesture: {hi_status}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    else:
        print("No hand detected in the image.")
        cv2.putText(image, "No hand detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the image with visualization
    cv2.imshow("Hand Gesture Recognition", image)
    cv2.waitKey(0)  # Wait for a key press to close the image window
    cv2.destroyAllWindows()

# Function to determine if the hand gesture is "Hi"
def detect_hi_gesture(hand_landmarks):
    finger_tips = [
        mp_hands.HandLandmark.THUMB_TIP,
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]
    finger_mcp = [
        mp_hands.HandLandmark.THUMB_MCP,
        mp_hands.HandLandmark.INDEX_FINGER_MCP,
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
        mp_hands.HandLandmark.RING_FINGER_MCP,
        mp_hands.HandLandmark.PINKY_MCP
    ]

    # Check if all fingertips are above their corresponding MCP joints
    palm_open = all(
        hand_landmarks.landmark[tip].y < hand_landmarks.landmark[mcp].y
        for tip, mcp in zip(finger_tips, finger_mcp)
    )
    return "Hi" if palm_open else "Not Hi"

# Example: Process a single image
# Update the path to the image you want to process
single_image_path = r'C:\Users\DELL\OneDrive\Desktop\SkillCrafttecnologies\Task4\leapGestRecog\frame_00_01_0001.png'
process_single_image(single_image_path)
