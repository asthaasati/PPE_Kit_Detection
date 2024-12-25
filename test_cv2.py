import cv2
print(cv2.getBuildInformation())


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access the webcam.")
else:
    print("Webcam opened successfully.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read from the webcam.")
            break

        cv2.imshow("Test Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
