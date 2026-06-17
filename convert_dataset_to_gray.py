import os
import shutil
import glob
import cv2
import yaml

def convert_to_3ch_grayscale(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_3ch_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(img_path, img_3ch_gray)
    return True

def main():
    src_dir = 'data_yolov8'
    dst_dir = 'data_yolov8_gray'
    
    if os.path.exists(dst_dir):
        print(f"Directory {dst_dir} already exists. Removing it first...")
        shutil.rmtree(dst_dir)
        
    print(f"Copying {src_dir} to {dst_dir}...")
    shutil.copytree(src_dir, dst_dir)
    
    print("Converting images to 3-channel grayscale...")
    for split in ['train', 'valid', 'test']:
        images_path = os.path.join(dst_dir, split, 'images', '*.jpg')
        images = glob.glob(images_path)
        print(f"Found {len(images)} images in {split} split.")
        for img_p in images:
            convert_to_3ch_grayscale(img_p)
            
    print("Fixing paths in data.yaml...")
    yaml_path = os.path.join(dst_dir, 'data.yaml')
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            
        data['train'] = 'train/images'
        data['val'] = 'valid/images'
        data['test'] = 'test/images'
        
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, sort_keys=False)
        print("Updated data.yaml successfully.")
    else:
        print("data.yaml not found!")
        
    print("Done!")

if __name__ == "__main__":
    main()
