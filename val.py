# import math
# import torch
# import torchvision
# from torchvision import transforms
# from torchvision.datasets import CocoDetection
# from PIL import Image
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import numpy as np

# def get_transform():

#     to_tensor = transforms.ToTensor()
    
#     def transform_fn(image, target):
#         image = to_tensor(image)
#         return image, target
    
#     return transform_fn

# def load_model(weights_path, num_classes):
 
    
#     model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    
#     in_features = model.roi_heads.box_predictor.cls_score.in_features
#     model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    
#     state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
#     model.load_state_dict(state_dict)
#     model.eval() 
#     return model

# def draw_boxes(ax, gt_boxes, predictions, label_names, threshold=0.5):
   

#     for bbox in gt_boxes:
#         x, y, w, h = bbox  
#         gt_rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor="green", facecolor="none")
#         ax.add_patch(gt_rect)
#         ax.text(x, y, "GT", fontsize=10, color="green", bbox=dict(facecolor="black", alpha=0.5))
    
    
#     for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
#         if score < threshold:
#             continue
#         x1, y1, x2, y2 = box.tolist()
#         pred_rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
#                                       linewidth=2, edgecolor="red", facecolor="none")
#         ax.add_patch(pred_rect)
#         label_name = label_names.get(label.item(), str(label.item()))
#         ax.text(x1, y1, f"{label_name}: {score:.2f}", fontsize=10, color="yellow",
#                 bbox=dict(facecolor='red', alpha=0.5))

# def main():

#     val_images_path = "/home/hp/Desktop/faster rcnn/dataset1/val"   
#     val_ann_file = "/home/hp/Desktop/faster rcnn/dataset1/val.json"    
#     weights_path = "/home/hp/Desktop/vas/fasterrcnn_resnet50_epoch_141.pth"  
    
    
#     num_classes = 2  # 
    
    
#     label_names = {
#         1: "hole",  
#     }
    
#     transform = get_transform()
#     val_dataset = CocoDetection(root=val_images_path, annFile=val_ann_file, transforms=transform)
    
#     model = load_model(weights_path, num_classes)
    
#     num_visualizations = 5  
#     cols = 3                
#     rows = math.ceil(num_visualizations / cols)
    
#     fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))

#     if isinstance(axes, np.ndarray):
#         axes = axes.flatten()
#     else:
#         axes = [axes]

#     for idx in range(num_visualizations):
        
#         image_tensor, target = val_dataset[idx]
        
        
#         with torch.no_grad():
#             predictions = model([image_tensor])[0]
        
        
#         img_np = image_tensor.mul(255).permute(1, 2, 0).byte().numpy()
#         pil_img = Image.fromarray(img_np)
        
        
#         gt_boxes = [ann["bbox"] for ann in target if "bbox" in ann]
        
        
#         ax = axes[idx]
#         ax.imshow(pil_img)
#         ax.set_title(f"Image {idx+1}")
#         ax.axis("off")
        
        
#         draw_boxes(ax, gt_boxes, predictions, label_names, threshold=0.5)
    
    
#     for j in range(num_visualizations, len(axes)):
#         axes[j].axis("off")
    
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     main()

import math
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import CocoDetection
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def get_transform():
    """
    Returns a function that takes an image and target, transforms the image to a tensor,
    and leaves the target unchanged.
    """
    to_tensor = transforms.ToTensor()
    
    def transform_fn(image, target):
        image = to_tensor(image)
        return image, target
    
    return transform_fn

def load_model(weights_path, num_classes):
    """
    Load a fine-tuned Faster R-CNN model with the correct number of classes.
    
    Args:
        weights_path (str): Path to the fine-tuned weights.
        num_classes (int): Number of classes (including background if applicable).
        
    Returns:
        model: The model loaded with fine-tuned weights in evaluation mode.
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()  # Set model to evaluation mode
    return model

def draw_boxes(ax, gt_boxes, predictions, label_names, threshold=0.5):
    """
    Draw ground truth and predicted bounding boxes on a given matplotlib Axes.
    
    Ground truth boxes are drawn in green and predicted boxes in red.
    
    Args:
        ax (matplotlib.axes.Axes): The subplot axes.
        gt_boxes (list): List of ground truth bounding boxes in [x, y, width, height] format.
        predictions (dict): Model predictions (expects keys 'boxes', 'labels', 'scores').
        label_names (dict): Mapping from label id to label name.
        threshold (float): Confidence threshold to filter predicted boxes.
    """
    # Draw ground truth boxes in green.
    for bbox in gt_boxes:
        x, y, w, h = bbox  # COCO format: [x, y, width, height]
        gt_rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor="green", facecolor="none")
        ax.add_patch(gt_rect)
        ax.text(x, y, "GT", fontsize=10, color="green", bbox=dict(facecolor="black", alpha=0.5))
    
    # Draw predicted boxes in red.
    for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
        if score < threshold:
            continue
        x1, y1, x2, y2 = box.tolist()
        pred_rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                      linewidth=2, edgecolor="red", facecolor="none")
        ax.add_patch(pred_rect)
        label_name = label_names.get(label.item(), str(label.item()))
        ax.text(x1, y1, f"{label_name}: {score:.2f}", fontsize=10, color="yellow",
                bbox=dict(facecolor='red', alpha=0.5))

def visualize_predictions(val_dataset, model, label_names, num_visualizations=5, cols=3):
    """
    Display a grid of images with their ground truth and predicted bounding boxes.
    """
    rows = math.ceil(num_visualizations / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]
    
    for idx in range(num_visualizations):
        image_tensor, target = val_dataset[idx]
        with torch.no_grad():
            predictions = model([image_tensor])[0]
        img_np = image_tensor.mul(255).permute(1, 2, 0).byte().numpy()
        pil_img = Image.fromarray(img_np)
        gt_boxes = [ann["bbox"] for ann in target if "bbox" in ann]
        ax = axes[idx]
        ax.imshow(pil_img)
        ax.set_title(f"Image {idx+1}")
        ax.axis("off")
        draw_boxes(ax, gt_boxes, predictions, label_names, threshold=0.5)
    
    for j in range(num_visualizations, len(axes)):
        axes[j].axis("off")
    
    plt.tight_layout()
    plt.show()

def count_successful_detections(val_dataset, model, target_label=1, threshold=0.5):
    """
    Loop over the validation dataset and count how many images have at least one detection
    with the given target_label (e.g. hole) and a score above the threshold.
    
    Args:
        val_dataset (CocoDetection): The validation dataset.
        model: The fine-tuned model.
        target_label (int): The label id to check (for hole, use 1).
        threshold (float): Score threshold for considering a detection valid.
    
    Returns:
        successful_count (int): Number of images with a successful detection.
        total_images (int): Total number of images processed.
    """
    successful_count = 0
    total_images = len(val_dataset)
    
    for idx in range(total_images):
        image_tensor, _ = val_dataset[idx]
        with torch.no_grad():
            predictions = model([image_tensor])[0]
        # Check if any prediction has the target label and a high enough score.
        detected = any((label.item() == target_label and score >= threshold) 
                       for label, score in zip(predictions['labels'], predictions['scores']))
        if detected:
            successful_count += 1
    return successful_count, total_images

def main():
    # -------------------------------
    # 1. Set Paths, Number of Classes, and Label Mapping
    # -------------------------------
    val_images_path = "/home/hp/Desktop/faster rcnn/dataset1/val"   # Folder containing validation images.
    val_ann_file = "/home/hp/Desktop/faster rcnn/dataset1/val.json"    # COCO-style annotation file.
    weights_path = "/home/hp/Desktop/vas/fasterrcnn_resnet50_epoch_141.pth"  # Fine-tuned model weights.
    
    num_classes = 2  # Adjust as necessary.
    
    # Mapping from label id to class name.
    label_names = {
        1: "hole",  # Label 1 corresponds to your object "hole".
    }
    
    # -------------------------------
    # 2. Define Transform and Load the Dataset
    # -------------------------------
    transform = get_transform()
    val_dataset = CocoDetection(root=val_images_path, annFile=val_ann_file, transforms=transform)
    
    # -------------------------------
    # 3. Load the Fine-Tuned Model
    # -------------------------------
    model = load_model(weights_path, num_classes)
    
    # -------------------------------
    # 4. Visualize Some Predictions (Optional)
    # -------------------------------
    visualize_predictions(val_dataset, model, label_names, num_visualizations=5, cols=3)
    
    # -------------------------------
    # 5. Count Successful "Hole" Detections in the Validation Set
    # -------------------------------
    # Here, we consider an image successful if it has at least one detection with label 1 ("hole")
    # and a confidence score of at least 0.5.
    target_label = 1  # hole
    score_threshold = 0.5
    successful_count, total_images = count_successful_detections(val_dataset, model, target_label, score_threshold)
    
    print("\nEvaluation of Successful 'Hole' Detections:")
    print(f"Out of {total_images} images, the model successfully detected a hole in {successful_count} images.")
    print(f"Success Rate: {successful_count/total_images*100:.2f}%")
    
if __name__ == "__main__":
    main()
