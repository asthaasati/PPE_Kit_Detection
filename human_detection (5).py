import cv2
import numpy as np
import os
import sys

# --- File Paths ---
weights_path = r"C:\Users\asati\Desktop\ppe kit detection\yolov3.weights"
config_path = r"C:\Users\asati\Desktop\ppe kit detection\yolov3 (2).cfg"
video_file = r"C:\Users\asati\Desktop\ppe kit detection\video2.mp4"  # Default video file

# --- Check for Required Files ---
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Error: YOLO weights file '{weights_path}' not found.")
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Error: YOLO config file '{config_path}' not found.")

# --- Load YOLO ---
print("Loading YOLO model...")
net = cv2.dnn.readNet(weights_path, config_path)

# Use GPU if available (optional: uncomment if GPU is enabled)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Load class names
coco_names_path = r"C:\Users\asati\Desktop\ppe kit detection\coco (1).names"
if not os.path.exists(coco_names_path):
    raise FileNotFoundError(f"Error: COCO class names file '{coco_names_path}' not found.")

with open(coco_names_path, "r") as file:
    classes = [line.strip() for line in file.readlines()]

# Get output layer names (compatible with new and old OpenCV versions)
layer_names = net.getLayerNames()
try:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
except AttributeError:
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# --- Function to Process Video ---
def process_video(video_source):
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        raise FileNotFoundError(f"Error: Could not open video source '{video_source}'.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        height, width, channels = frame.shape

        # Prepare the frame for the YOLO model
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Lists for detected bounding boxes, confidences, and class IDs
        boxes = []
        confidences = []
        class_ids = []

        # Process each detection
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # Only detect 'person' with high confidence
                if confidence > 0.5 and class_id == classes.index("person"):
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Calculate bounding box coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply Non-Maximum Suppression (NMS)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Draw bounding boxes and labels on the frame
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = (0, 255, 0)  # Green color for 'person'
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display the video frame
        cv2.imshow("Human Detection", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

# --- Main Script ---
def main():
    print("Select an option:")
    print("1. Process a video file")
    print("2. Use webcam")
    choice = input("Enter your choice (1/2): ").strip()

    if choice == "1":
        video_path = input("Enter the path to the video file: ").strip()
        if not os.path.exists(video_path):
            print(f"Error: Video file '{video_path}' not found.")
            return
        process_video(video_path)
    elif choice == "2":
        print("Starting webcam...")
        process_video(0)  # 0 is the default device index for the webcam
    else:
        print("Invalid choice. Exiting.")

    cv2.destroyAllWindows()
    print("Processing complete.")

if __name__ == "__main__":
    main()