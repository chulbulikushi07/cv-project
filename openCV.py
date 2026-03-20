import cv2
from ultralytics import YOLO
# we measure fps = 1/(time taken to process one frame)
import time 

#Load Yolo model
#Yolov8 = model version, n = nano(small and fast), .pt = pytorch model file 
model = YOLO("yolov8n.pt")

#opens webcam , 0 is default camera(1 and 2 might be other camera but 0 for the time is the default camera)
cam = cv2.VideoCapture(0)

#check if camera opened succesfully (if another camera is open that might cause issues)
if not cam.isOpened():
    print("Error: Could not open webcam.")
    exit()

prev_time = 0

while True:
    #time.time() gives current time in seconds.
    start_time = time.time()
    
    #this grabs one frame from the camera and returns it with a boolean value(t/f)
    #frame is a NumPy array of shape (height, width, channels)
    #channels = 3(BGR)
    #openCV uses BGR and not RGB cuz tradition.
    ret, frame = cam.read()

    if not ret:
        print("Error: Can't receive frame.")
        break
    #results is a list of detection results,we only passed one image so we use results[0].
    results = model(frame)

    # Draw bounding boxes + Draw labels + Draws confidence scores.
    annotated_frame = results[0].plot()

    #calculating fps ,end time - start time = total time for one loop iteration.
    end_time = time.time()
    fps = 1/(end_time - start_time)

    #putting fps text on screen
    #This draws text on the image.

    #Parameters:

    #Image
    #Text
    #Position (x, y)
    #Font
    #Size
    #Color (B, G, R)
    #Thickness
    
    cv2.putText(
        annotated_frame,
        f"FPS: {int(fps)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Display the processed frame in a window
    #first arguement shows the windows title and second arguement is the image array
    cv2.imshow("YOLO Detection", annotated_frame)
    
    #waitkey(1) basically means that this waits one milisecond to press ,&0xFF standardizes the key code
    #ord('q') is basically the ASCII value of q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#releases webcam hardware cuz whithout this camera might stay locked 
cam.release()

#closes all openCV windows to prevent frozen windows basically
cv2.destroyAllWindows()