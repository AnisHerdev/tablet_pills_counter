from ultralytics import YOLO
import cv2
import os

# 1. Load the exported TFLite model
# Note: You will need to replace 'best_int8.tflite' with the new model exported by the grayscale notebook
model = YOLO('best_int8.tflite', task='detect')

# 2. Point it to a test image
image_path = '20260617_021150.jpg'

if not os.path.exists(image_path):
    print(f"Could not find test image at {image_path}")
    print("Please update the 'image_path' variable to a valid image path!")
    exit(1)

print(f"Running inference on {image_path}...\n")

# 3. Load image and convert to 3-channel grayscale exactly like training
img = cv2.imread(image_path)
if img is None:
    print(f"Failed to load image at {image_path}")
    exit(1)
    
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_3ch_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# 4. Run inference (pass the processed numpy array directly)
results = model.predict(img_3ch_gray, conf=0.25)

# 5. Extract and print the counting logic
for r in results:
    classes = r.boxes.cls.cpu().numpy()
    
    empty_count = sum(classes == 0)
    fill_count = sum(classes == 1)
    
    print("\n" + "="*30)
    print("--- TABLET COUNT RESULTS ---")
    print("="*30)
    print(f"💊 Pills Present (Fill): {fill_count}")
    print(f"⭕ Empty Cavities:       {empty_count}")
    print(f"📦 Total Sheet Capacity: {empty_count + fill_count}")
    print("="*30 + "\n")

    print("Opening image viewer with bounding boxes... (Press any key or close window to exit)")
    r.show()
