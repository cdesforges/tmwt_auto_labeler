import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

cap = cv2.VideoCapture("media/SV_10MWRT_string.MOV")
if not cap.isOpened():
    print("Cannot open video!")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
print("FPS:", fps)

# calculate frame delay
delay = int(1000 / fps) if fps > 0 else 30

# set up mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose



while True:
    # get frame
    ret, frame = cap.read()
    if not ret:
        break

    # edit frame
    cv2.putText(
        frame, "Hello", (50,50),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    # show the frame
    cv2.imshow("Video", frame)

    # wait for input
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()