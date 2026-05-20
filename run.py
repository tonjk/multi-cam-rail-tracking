import cv2
from src.detector import Detector

# read video and display in a while loop

# cap = cv2.VideoCapture("videos/Thai_train_passing.mp4")
cap = cv2.VideoCapture("videos/Cars_driving_railway_crossing.mp4")
detector = Detector()

while True:
    if not cap.isOpened():
        print("Unable to open the video file.")
        break

    # Get video properties for saving (optional use later)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    ret, frame = cap.read()
    if not ret:
        print("End of video stream or error reading frame.")
        break
        
    # Run predictions and apply custom premium visualization
    results = detector.predict(frame)
    annotated_frame = detector.draw_results(frame, results)

    cv2.imshow("frame", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Resource cleanup
cap.release()
cv2.destroyAllWindows()
