# dataset_export.py (dataset export functionality)
import os
import shutil
import yaml
import random
from sklearn.model_selection import train_test_split
from PIL import Image

def export_yolo_dataset(annotations, export_dir):
    # Create directory structure
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(export_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(export_dir, split, 'labels'), exist_ok=True)
    
    # Prepare data
    image_paths = list(annotations.keys())
    random.shuffle(image_paths)
    
    # Split dataset
    train_val, test = train_test_split(image_paths, test_size=0.05, random_state=42)
    train, val = train_test_split(train_val, test_size=0.15/0.95, random_state=42)
    
    # Create YAML config
    data_yaml = {
        'names': ['object'],
        'nc': 1,
        'test': 'test/images',
        'train': 'train/images',
        'val': 'val/images'
    }
    
    # Process splits
    for split_name, split_data in zip(splits, [train, val, test]):
        for image_path in split_data:
            # Copy image
            img_name = os.path.basename(image_path)
            dest_img_path = os.path.join(export_dir, split_name, 'images', img_name)
            shutil.copy(image_path, dest_img_path)
            
            # Create annotation file
            txt_path = os.path.join(export_dir, split_name, 'labels', 
                                  os.path.splitext(img_name)[0] + '.txt')
            with open(txt_path, 'w') as file:
                for mask_data in annotations[image_path]:
                    polygon = mask_data['polygon']
                    bbox = mask_data['bbox']
                    
                    # Normalize coordinates
                    img_width, img_height = Image.open(image_path).size
                    normalized_polygon = [
                        x / img_width if i % 2 == 0 else y / img_height 
                        for i, (x, y) in enumerate(polygon)
                    ]
                    normalized_bbox = [
                        bbox[0] / img_width,
                        bbox[1] / img_height,
                        (bbox[2] - bbox[0]) / img_width,
                        (bbox[3] - bbox[1]) / img_height
                    ]
                    
                    # Write to file (YOLOv8 segmentation format)
                    class_id = 0  # Assuming single class
                    yolo_line = [f"{class_id}"] + [f"{coord:.6f}" for coord in normalized_polygon]
                    file.write(" ".join(yolo_line) + "\n")
    
    # Save data.yaml
    yaml_path = os.path.join(export_dir, 'data.yaml')
    with open(yaml_path, 'w') as file:
        yaml.dump(data_yaml, file)