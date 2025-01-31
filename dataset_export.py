# dataset_export.py
import os
import shutil
import yaml
import random
from sklearn.model_selection import train_test_split
from PIL import Image

def export_yolo_dataset(annotations, export_dir, class_names=None):
    # Create directories
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
    if class_names:
        names = [class_names[cid] for cid in sorted(class_names)]
        nc = len(names)
    else:
        names = ['object']
        nc = 1
    
    data_yaml = {
        'names': names,
        'nc': nc,
        'test': 'test/images',
        'train': 'train/images',
        'val': 'val/images'
    }
    
    # Process splits
    for split_name, split_data in zip(splits, [train, val, test]):
        for image_path in split_data:
            img_name = os.path.basename(image_path)
            dest_img = os.path.join(export_dir, split_name, 'images', img_name)
            shutil.copy(image_path, dest_img)
            
            txt_path = os.path.join(export_dir, split_name, 'labels', 
                                  os.path.splitext(img_name)[0] + '.txt')
            with open(txt_path, 'w') as f:
                img = Image.open(image_path)
                w, h = img.size
                for ann in annotations[image_path]:
                    poly = ann['polygon']
                    class_id = ann['class_id']
                    normalized = [x/w if i%2==0 else y/h for i, (x,y) in enumerate(poly)]
                    f.write(f"{class_id} " + " ".join(f"{c:.6f}" for c in normalized) + "\n")
    
    # Save YAML
    with open(os.path.join(export_dir, 'data.yaml'), 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)