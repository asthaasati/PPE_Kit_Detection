import cv2
import numpy as np

# Load pre-trained model (YOLO)
net = cv2.dnn.readNet(r"C:\Users\asati\Desktop\ppe kit detection\yolov3.weights", 
                     r"C:\Users\asati\Desktop\ppe kit detection\yolov3 (2).cfg")

layer_names = net.getLayerNames()
# Fix: Directly access layer indices in net.getUnconnectedOutLayers()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Open the webcam
cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Prepare the image for prediction (YOLO example)
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        # Process outputs and draw bounding boxes for detected PPE
        for out in outputs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # Confidence threshold
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    w = int(detection[2] * frame.shape[1])
                    h = int(detection[3] * frame.shape[0])

                    # Draw the rectangle around detected PPE
                    cv2.rectangle(frame, (center_x, center_y), (center_x + w, center_y + h), (0, 255, 0), 2)
                    label = f"Object {confidence:.2f}"
                    cv2.putText(frame, label, (center_x, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("PPE Detection", frame)

        # Exit loop gracefully on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("Program interrupted. Exiting...")
finally:
    cap.release()
    cv2.destroyAllWindows()
