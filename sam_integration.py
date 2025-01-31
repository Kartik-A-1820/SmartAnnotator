# sam_integration.py (SAM model handling)
import torch
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
from segment_anything.utils.transforms import ResizeLongestSide

def load_sam_model(device="cuda" if torch.cuda.is_available() else "cpu"):
    if device == "cuda":
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"
    else:
        sam_checkpoint = "sam_vit_b_01ec64.pth"
        model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    return SamPredictor(sam)

def generate_masks(predictor, image_array, input_point, input_label=np.array([1])):

    try:
        predictor.set_image(image_array)
        
        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
        print(f"Masks: {len(masks)}")
        return masks
    
    except:
        # Transform image to SAM's expected format
        transform = ResizeLongestSide(1024)
        input_image = transform.apply_image(image_array)
        input_image_torch = torch.as_tensor(input_image, device=predictor.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        
        # Preprocess image
        predictor.set_torch_image(input_image_torch, image_array.shape[:2])
        
        # Transform coordinates
        input_point = transform.apply_coords(input_point, image_array.shape[:2])
        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=np.array([1]),
            multimask_output=False,
        )
        print(f"Masks: {len(masks)}")
        return masks
