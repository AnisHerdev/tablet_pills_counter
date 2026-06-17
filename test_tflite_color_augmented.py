from ultralytics import YOLO
import cv2
import os

# 1. Load the exported TFLite model
# Note: You will need to replace 'best_int8.tflite' with the new model exported by the augmented notebook
model = YOLO('best_int8_color_augumented.tflite', task='detect')

# 2. Point it to a test image
image_path = '20260617_021150.jpg'

if not os.path.exists(image_path):
    print(f"Could not find test image at {image_path}")
    print("Please update the 'image_path' variable to a valid image path!")
    exit(1)

print(f"Running inference on {image_path}...\n")

# 3. Run inference (no preprocessing needed, the model handles color directly)
results = model.predict(image_path, conf=0.25)

# 4. Extract and print the counting logic
for r in results:
    classes = r.boxes.cls.cpu().numpy()
    
    empty_count = sum(classes == 0)
    full_count = sum(classes == 1)
    half_count = sum(classes == 2)
    
    print("\n" + "="*30)
    print("--- TABLET COUNT RESULTS ---")
    print("="*30)
    print(f"💊 Full Pills:           {full_count}")
    print(f"🌙 Half Pills:           {half_count}")
    print(f"⭕ Empty Cavities:       {empty_count}")
    print(f"📦 Total Sheet Capacity: {empty_count + full_count + half_count}")
    print("="*30 + "\n")

    print("Opening image viewer with bounding boxes... (Press any key or close window to exit)")
    r.show()
