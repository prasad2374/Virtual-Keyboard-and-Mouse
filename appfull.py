import cv2
import mediapipe as mp
import numpy as np
import pygame
import tkinter as tk
from threading import Thread
import pyautogui
import time
# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption('Eye Tracking Virtual Keyboard')

# Define keyboard layout
keyboard = [
    ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
    ['H', 'I', 'J', 'K', 'L', 'M', 'N'],
    ['O', 'P', 'Q', 'R', 'S', 'T', 'U'],
    ['V', 'W', 'X', 'Y', 'Z', 'SS', 'BS']
]

# Initialize variables
text = ""
last_blink_time = pygame.time.get_ticks()
blink_detected = False

# Helper function to draw keyboard
def draw_keyboard():
    for i, row in enumerate(keyboard):
        for j, key in enumerate(row):
            pygame.draw.rect(screen, (255, 255, 255), (j * 100, i * 100, 100, 100), 2)
            font = pygame.font.Font(None, 74)
            text_surface = font.render(key, True, (255, 255, 255))
            screen.blit(text_surface, (j * 100 + 10, i * 100 + 10))

# Function to calculate gaze direction
def calculate_gaze_direction(landmarks):
    left_eye_indices = [33, 160, 158, 133, 153, 144]  # Example indices for left eye
    right_eye_indices = [362, 263, 249, 390, 373, 380]  # Example indices for right eye

    left_eye_points = np.array([(landmarks[i].x, landmarks[i].y) for i in left_eye_indices])
    right_eye_points = np.array([(landmarks[i].x, landmarks[i].y) for i in right_eye_indices])
    
    left_eye_center = np.mean(left_eye_points, axis=0)
    right_eye_center = np.mean(right_eye_points, axis=0)
    
    gaze_x = int((left_eye_center[0] + right_eye_center[0]) * screen.get_width())
    gaze_y = int((left_eye_center[1] + right_eye_center[1]) * screen.get_height())

    return gaze_x, gaze_y

# Function to detect blink
def detect_blink(eye_landmarks):
    EAR_THRESHOLD = 0.2  # Eye Aspect Ratio threshold for blink detection
    EAR = calculate_ear(eye_landmarks)
    return EAR < EAR_THRESHOLD

# Function to calculate Eye Aspect Ratio (EAR)
def calculate_ear(eye_landmarks):
    eye_points = np.array([[lm.x, lm.y] for lm in eye_landmarks])
    
    A = np.linalg.norm(eye_points[1] - eye_points[5])  # Vertical distance between landmarks
    B = np.linalg.norm(eye_points[2] - eye_points[4])  # Vertical distance between landmarks
    C = np.linalg.norm(eye_points[0] - eye_points[3])  # Horizontal distance between landmarks
    
    EAR = (A + B) / (2.0 * C)
    return EAR

# Eye Typing Functionality
def eye_typing():
    global text, last_blink_time
    
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                left_eye_landmarks = [landmarks[i] for i in [33, 160, 158, 133, 153, 144]]
                right_eye_landmarks = [landmarks[i] for i in [362, 263, 249, 390, 373, 380]]
                
                if detect_blink(left_eye_landmarks) or detect_blink(right_eye_landmarks):
                    blink_time = pygame.time.get_ticks()
                    if blink_time - last_blink_time > 500:
                        last_blink_time = blink_time
                        cursor_x, cursor_y = calculate_gaze_direction(landmarks)
                        key_x = cursor_x // 100
                        key_y = cursor_y // 100
                        if 0 <= key_x < len(keyboard[0]) and 0 <= key_y < len(keyboard):
                            key = keyboard[key_y][key_x]
                            if key == 'SS':
                                text += ' '
                            elif key == 'BS':
                                text = text[:-1]
                            else:
                                text += key

                gaze_x, gaze_y = calculate_gaze_direction(landmarks)
                gaze_x = min(max(gaze_x, 0), screen.get_width() - 1)
                gaze_y = min(max(gaze_y, 0), screen.get_height() - 1)
                
                screen.fill((0, 0, 0))
                draw_keyboard()
                pygame.draw.circle(screen, (0, 255, 0), (gaze_x, gaze_y), 10)
                
                font = pygame.font.Font(None, 74)
                text_surface = font.render(text, True, (255, 255, 255))
                screen.blit(text_surface, (20, screen.get_height() - 100))
                
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                pygame.quit()
                return

        cv2.imshow('Eye Tracking', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        pygame.display.flip()

    cap.release()
    cv2.destroyAllWindows()


def virtual_keyboard():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1)
    mp_draw = mp.solutions.drawing_utils

    # Define the full virtual keyboard layout with increased box size and additional keys
    keyboard_keys = [
        ('Q', (70, 150)), ('W', (170, 150)), ('E', (270, 150)), ('R', (370, 150)), 
        ('T', (470, 150)), ('Y', (570, 150)), ('U', (670, 150)), ('I', (770, 150)), 
        ('O', (870, 150)), ('P', (970, 150)), ('A', (120, 250)), ('S', (220, 250)), 
        ('D', (320, 250)), ('F', (420, 250)), ('G', (520, 250)), ('H', (620, 250)), 
        ('J', (720, 250)), ('K', (820, 250)), ('L', (920, 250)), ('Z', (170, 350)), 
        ('X', (270, 350)), ('C', (370, 350)), ('V', (470, 350)), ('B', (570, 350)), 
        ('N', (670, 350)), ('M', (770, 350)), ('Space', (470, 450)), ('Backspace', (970, 350))
    ]

    def draw_keyboard(img):
        for key, pos in keyboard_keys:
            cv2.rectangle(img, (pos[0]-50, pos[1]-50), (pos[0]+50, pos[1]+50), (255, 0, 0), 3)
            cv2.putText(img, key, (pos[0]-30, pos[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    def is_pinch(hand_landmarks):
        index_finger_tip = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x, 
                                     hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y])
        thumb_tip = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x, 
                              hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y])
        distance = np.linalg.norm(index_finger_tip - thumb_tip)
        return distance < 0.05

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)
        
        draw_keyboard(frame)
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                height, width, _ = frame.shape
                x = int(index_finger_tip.x * width)
                y = int(index_finger_tip.y * height)
                
                cv2.circle(frame, (x, y), 10, (0, 255, 0), cv2.FILLED)
                
                if is_pinch(hand_landmarks):
                    for key, pos in keyboard_keys:
                        if abs(x - pos[0]) < 50 and abs(y - pos[1]) < 50:
                            cv2.putText(frame, key, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            if key == 'Space':
                                pyautogui.press('space')
                            elif key == 'Backspace':
                                pyautogui.press('backspace')
                            else:
                                pyautogui.press(key.lower())
                            time.sleep(0.5)  # Add delay after a key is pressed
                            break 
        
        cv2.imshow('Virtual Keyboard', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# Tkinter Interface
def start_virtual_keyboard():
    Thread(target=virtual_keyboard).start()


def start_eye_typing():
    Thread(target=eye_typing).start()

# Create the main window
root = tk.Tk()
root.title("Select Mode")
root.geometry("300x200")

# Create and place buttons
btn_virtual_keyboard = tk.Button(root, text="Virtual Keyboard", command=start_virtual_keyboard)
btn_virtual_keyboard.pack(pady=10)

btn_eye_typing = tk.Button(root, text="Eye Typing", command=start_eye_typing)
btn_eye_typing.pack(pady=10)

# Start the Tkinter main loop
root.mainloop()
