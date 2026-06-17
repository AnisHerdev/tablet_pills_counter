import cv2
import numpy as np
from ultralytics import YOLO

def main():
    # Load the TFLite model
    print("Loading model...")
    model = YOLO('best_int8_20.tflite', task='detect')
    print("Model loaded successfully!")

    # Initialize the camera (0 is usually the default laptop camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Create a dummy function for the trackbars
    def nothing(x):
        pass

    # Create a window for the live feed
    window_name = 'Live Pill Counter'
    cv2.namedWindow(window_name)

    # Create trackbars for "knobs"
    # 1. Confidence Threshold (0-100%, maps to 0.0-1.0)
    cv2.createTrackbar('Confidence %', window_name, 25, 100, nothing)
    
    # 2. IoU Threshold (0-100%, maps to 0.0-1.0). 
    # This controls Non-Maximum Suppression (overlap handling). 
    # Lower value means stricter overlap removal (keeps the bigger/higher confidence one).
    cv2.createTrackbar('Overlap (IoU) %', window_name, 45, 100, nothing)

    print("\n--- Live Pill Counter Started ---")
    print("Adjust the trackbars on the window to tune the model in real-time.")
    print("Press 'q' inside the video window to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        try:
            # Check if window was closed via 'X' button
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
                
            # Read the current values from the trackbars
            conf_percent = cv2.getTrackbarPos('Confidence %', window_name)
            iou_percent = cv2.getTrackbarPos('Overlap (IoU) %', window_name)
        except cv2.error:
            # Window was closed
            break

        # Convert percentages to float (0.0 to 1.0)
        conf = max(0.01, conf_percent / 100.0) # Avoid exactly 0
        iou = max(0.01, iou_percent / 100.0)   # Avoid exactly 0

        # Run inference on the current frame
        # We pass the conf and iou parameters dynamically!
        results = model.predict(frame, conf=conf, iou=iou, verbose=False)
        
        # We only pass one frame, so we only get one result object
        result = results[0]
        
        # Let Ultralytics draw the bounding boxes and labels for us
        annotated_frame = result.plot()
        
        # Extract counts based on class IDs
        classes = result.boxes.cls.cpu().numpy()
        empty_count = sum(classes == 0)
        full_count = sum(classes == 1)
        half_count = sum(classes == 2)
        
        # Add a dark background rectangle for text readability
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (0, 0), (280, 160), (0, 0, 0), -1)
        annotated_frame = cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0)
        
        # Display the counts as text on the live video feed
        y_offset = 30
        cv2.putText(annotated_frame, f'Full Pills: {full_count}', (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f'Half Pills: {half_count}', (10, y_offset + 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(annotated_frame, f'Empty: {empty_count}', (10, y_offset + 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        total_pills = full_count + half_count
        cv2.putText(annotated_frame, f'Total Pills: {total_pills}', (10, y_offset + 115), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Show the frame
        try:
            cv2.imshow(window_name, annotated_frame)
        except cv2.error:
            break

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
