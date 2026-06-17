import torch
from ultralytics import YOLO

# ==========================================
# TRAINING CONFIGURATION
# Modify these parameters as needed before running
# ==========================================
EPOCHS = 20
IMG_SIZE = 640
DATA_YAML_PATH = 'data_yolov8_gray/data.yaml'
BATCH_SIZE = 4          # lower this further if CUDA OOM warnings continue
WORKERS = 0              # reduce data-loader overhead on small-memory machines
# ==========================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp = device == 'cuda'
    print(f'Using device: {device}')
    print(f'Mixed precision enabled: {use_amp}')

    # 1. Load a pre-trained YOLOv8 Nano model
    model = YOLO('yolov8n.pt')

    # 2. Train the model
    print("Starting training...")
    results = model.train(
        data=DATA_YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        device=device,
        amp=use_amp
    )
    print("Training completed!")

    # 3. Export the model to INT8 TFLite
    print("Exporting model to INT8 TFLite format...")
    export_path = model.export(
        format='tflite',
        int8=True,
        data=DATA_YAML_PATH,
        imgsz=IMG_SIZE
    )
    print(f"Export completed! TFLite model saved at: {export_path}")

if __name__ == '__main__':
    main()
