import torch
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def main():
    image_path = "/home/hp/Faster R-CNN/dataset1/val/8213_cam1_782286_png.rf.5ad4e9f8084595098d3d934af4bb1cc8.jpg"             
    fine_tuned_weights_path = "/home/hp/Desktop/vas/fasterrcnn_resnet50_epoch_141.pth" 

    
    image = Image.open(image_path).convert("RGB")

    
    transform = transforms.Compose([
        transforms.ToTensor(),  
    ])
    image_tensor = transform(image)

    num_classes = 2  

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    
    state_dict = torch.load(fine_tuned_weights_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval() 

    with torch.no_grad():  
        predictions = model([image_tensor])

    pred = predictions[0]
    boxes = pred['boxes']   
    labels = pred['labels'] 
    scores = pred['scores'] 

    confidence_threshold = 0.5  
    keep = scores >= confidence_threshold
    filtered_boxes = boxes[keep]
    filtered_labels = labels[keep]
    filtered_scores = scores[keep]
    label_names = {1:"hole"}

    
    fig, axis = plt.subplots(1, figsize=(12, 9))
    axis.imshow(image)

    for box, label,score in zip(filtered_boxes,filtered_labels, filtered_scores):
        x1, y1, x2, y2 = box.tolist()
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        axis.add_patch(rect)
        label_name = label_names.get(label.item(), str(label.item()))
        axis.text(x1, y1, f":{label_name}: {score:.2f}", color="yellow", fontsize=12, backgroundcolor="red")

    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
