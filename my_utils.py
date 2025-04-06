# basic python libraries
import os
import random
import numpy as np
import pandas as pd
import cv2
# for ignoring warnings
import warnings
warnings.filterwarnings('ignore')

# openCV


# xml library for parsing xml files
from xml.etree import ElementTree as et

# matplotlib for visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# torchvision libraries
import torch
import torchvision
from torchvision import transforms as torchtrans
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# helper libraries
from engine import train_one_epoch, evaluate

import utils
# import transforms as T

# for image augmentations
# import albumentations as A
# from albumentations.pytorch.transforms import ToTensorV2
import json
import os
import cv2
import torch
import numpy as np
import xml.etree.ElementTree as et
from torch.utils.data import Dataset

import os
import cv2
import torch
import numpy as np
import xml.etree.ElementTree as et
from torch.utils.data import Dataset

import os
import cv2
import torch
import numpy as np
import xml.etree.ElementTree as et
from torch.utils.data import Dataset

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as et

class FruitImagesDataset(Dataset):
    def __init__(self, train_dir, width, height, transforms=None):
        self.transforms = transforms
        self.train_dir = train_dir
        self.height = height
        self.width = width

        # Load image filenames with .jpg extension
        self.imgs = [img for img in sorted(os.listdir(train_dir)) if img.endswith('.jpg')]

        # Map each image filename to a consistent integer ID
        self.image_id_map = {img: idx for idx, img in enumerate(self.imgs)}

        # Define class names
        self.classes = ['__background__', 'SC', 'SN']  # Background: 0, SC: 1, SN: 2
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        image_path = os.path.join(self.train_dir, img_name)

        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError(f"Error loading image: {image_path}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_res = cv2.resize(img_rgb, (self.width, self.height), cv2.INTER_AREA)
        img_res /= 255.0

        # Use consistent image_id
        image_id = self.image_id_map[img_name]

        # Load annotations
        annot_path = os.path.join(self.train_dir, img_name.replace('.jpg', '.jpg.xml'))

        boxes, labels = [], []
        if os.path.exists(annot_path):
            tree = et.parse(annot_path)
            root = tree.getroot()

            original_w, original_h = img.shape[1], img.shape[0]

            for obj in root.findall('object'):
                label_text = obj.find('name').text
                if label_text not in self.class_to_idx:
                    continue

                label = self.class_to_idx[label_text]
                labels.append(label)

                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)

                xmin = (xmin / original_w) * self.width
                xmax = (xmax / original_w) * self.width
                ymin = (ymin / original_h) * self.height
                ymax = (ymax / original_h) * self.height

                boxes.append([xmin, ymin, xmax, ymax])

        print(f"[INFO] {len(boxes)} box(es) found for image: {img_name}")

        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([image_id]),
            'area': area,
            'iscrowd': iscrowd
        }

        if self.transforms:
            sample = self.transforms(image=img_res, bboxes=boxes.tolist(), labels=labels.tolist())
            img_res = sample['image']
            target['boxes'] = torch.tensor(sample['bboxes'], dtype=torch.float32)
        else:
            img_res = torch.tensor(img_res).permute(2, 0, 1).float()

        return img_res, target

    def __len__(self):
        return len(self.imgs)


def count_classes(dataloader, class_names):
    counts = {name: 0 for name in class_names[1:]}  # skip background
    for _, targets in dataloader:
        for target in targets:
            labels = target['labels'].tolist()
            for lbl in labels:
                class_name = class_names[lbl]
                if class_name in counts:
                    counts[class_name] += 1
    return counts

from coco_eval import CocoEvaluator  # Your custom evaluator
from pycocotools.coco import COCO
import torch

def evaluate(model, dataloader, coco_gt, device):
    model.eval()

    # Debugging: Inspect the first validation sample
    val_dataset = dataloader.dataset
    print("\n🔍 [DEBUG] Checking first validation sample prediction...")
    
    for img, target in val_dataset:
        img = img.to(device).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            output = model(img)[0]
        
        print("Predicted boxes:", output['boxes'])
        print("Scores:", output['scores'])
        print("Labels:", output['labels'])
        break  # Exit after first sample

    results = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = list(img.to(device) for img in images)
            outputs = model(images)

            for target, output in zip(targets, outputs):
                image_id = int(target["image_id"].item())

                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()

                for box, score, label in zip(boxes, scores, labels):
                    x_min, y_min, x_max, y_max = box
                    width = x_max - x_min
                    height = y_max - y_min

                    results.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [x_min, y_min, width, height],
                        "score": float(score)
                    })

    # 🔥 Create evaluator and update with flat list of results
    coco_evaluator = CocoEvaluator(coco_gt, iou_types=["bbox"])
    coco_evaluator.update(results)
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    return coco_evaluator




from pycocotools.coco import COCO
from coco_eval import CocoEvaluator  # if custom class used

import torch
import numpy as np
from pycocotools.coco import COCO

import torch
import numpy as np
from pycocotools.coco import COCO

def train_model(model, dataloader_train, dataloader_val, optimizer, lr_scheduler, device, num_epochs, save_path, patience=10):
    model.to(device)
    best_val_loss = float("inf")
    best_model_path = f"{save_path}/best_model.pth"
    
    val_losses = []
    val_maps = []
    no_improvement_epochs = 0  # Track epochs without improvement

    class_names = ['__background__', 'SC', 'SN']
    coco_gt = COCO("/work/shared/ngmm/scripts/Beyza_Zayim/datachon/output/coco_export/instances_train_coco.json")

    print("\n🔍 [DEBUG] Checking validation dataset labels (first 5 samples)...")
    val_dataset = dataloader_val.dataset

    for i in range(min(5, len(val_dataset))):
        _, target = val_dataset[i]
        labels = target["labels"].tolist()
        print(f"Sample {i} labels:", labels)
        if 0 in labels:
            print(f"⚠️ WARNING: Background class label (0) found in sample {i}. This might break COCO evaluation!")

    for epoch in range(num_epochs):
        print(f"\n[INFO] Epoch {epoch + 1}/{num_epochs} ---------------------")

        train_one_epoch(model, optimizer, dataloader_train, device, epoch, print_freq=10)
        lr_scheduler.step()

        # Validate and compute COCO metrics
        val_stats = evaluate(model, dataloader_val, coco_gt, device=device)
        val_map = val_stats.coco_eval['bbox'].stats[0]
        val_maps.append(val_map)

        # Compute validation loss
        val_loss = 0.0
        model.train()  # Ensure model returns loss during validation
        with torch.no_grad():
            for images, targets in dataloader_val:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                output = model(images, targets)  # Should return a dict of losses

                if isinstance(output, dict):  # Ensure it contains loss values
                    total_loss = sum(output.values())
                    val_loss += total_loss.item()
                else:
                    print("⚠️ Warning: Model output is not a dictionary! Skipping loss calculation.")

        val_loss /= len(dataloader_val)
        val_losses.append(val_loss)

        print(f"✅ Epoch {epoch + 1} Complete | Val Loss: {val_loss:.4f} | Val mAP: {val_map:.4f}")

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_epochs = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"🔥 New Best Model Saved! Val Loss: {val_loss:.4f}")
        else:
            no_improvement_epochs += 1
            print(f"⚠️ No improvement for {no_improvement_epochs}/{patience} epochs.")

        # Early stopping condition
        if no_improvement_epochs >= patience:
            print(f"⏹️ Early stopping triggered after {patience} epochs without improvement.")
            break

    print("\n🎉 Training Complete.")
    print("Validation Losses:", val_losses)
    print("Validation mAPs:", val_maps)
    print(f"🏆 Best model saved to {best_model_path} with Val Loss: {best_val_loss:.4f}")



import os
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm

def convert_voc_to_coco(voc_folder, output_json_path, class_names):
    annotation_files = sorted([f for f in os.listdir(voc_folder) if f.endswith('.xml')])

    categories = []
    for idx, name in enumerate(class_names):
        categories.append({
            "id": idx + 1,
            "name": name,
            "supercategory": "object"
        })
    class_to_id = {name: idx for idx, name in enumerate(class_names, start=1)}


    images = []
    annotations = []
    annotation_id = 1

    for image_id, xml_file in enumerate(tqdm(annotation_files)):
        xml_path = os.path.join(voc_folder, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = root.find('filename').text
        width = int(root.find('size/width').text)
        height = int(root.find('size/height').text)

        images.append({
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height
        })

        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in class_names:
                continue

            label = class_to_id[class_name]

            bndbox = obj.find('bndbox')
            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))
            width_box = xmax - xmin
            height_box = ymax - ymin

            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": label,
                "bbox": [xmin, ymin, width_box, height_box],
                "area": width_box * height_box,
                "iscrowd": 0
            })
            annotation_id += 1

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_json_path, 'w') as f:
        json.dump(coco_format, f, indent=4)

    print(f"✅ COCO annotation file saved to: {output_json_path}")
import torch
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

def plot_predictions(image_tensor, boxes, labels, scores, class_names, score_threshold=0.5):
    """
    Plots predictions on an image with bounding boxes, labels, and scores.
    """
    image = to_pil_image(image_tensor.cpu())
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)

    for box, label, score in zip(boxes, labels, scores):
        if score < score_threshold:
            continue

        xmin, ymin, xmax, ymax = box
        width, height = xmax - xmin, ymax - ymin
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2,
                                 edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        label_text = f"{class_names[label]}: {score:.2f}"
        ax.text(xmin, ymin - 5, label_text, color='white', backgroundcolor='green', fontsize=10)

    plt.axis("off")
    plt.show()

def test_model(model, dataloader_test, device, save_results_path, plot=True, score_threshold=0.5):
    """
    Runs inference on the test dataset, saves predictions, and optionally plots them.
    """
    class_names = ['__background__', 'SC', 'SN']
    model.eval()
    results = []

    with torch.no_grad():
        for i, (images, _) in tqdm(enumerate(dataloader_test), total=len(dataloader_test)):
            images = [img.to(device) for img in images]
            predictions = model(images)

            for img_idx in range(len(images)):
                pred_labels = predictions[img_idx]["labels"].cpu().tolist()
                pred_boxes = predictions[img_idx]["boxes"].cpu().tolist()
                pred_scores = predictions[img_idx]["scores"].cpu().tolist()

                result = {
                    "image_id": i * len(images) + img_idx,  # ensure unique IDs
                    "pred_boxes": pred_boxes,
                    "pred_labels": pred_labels,
                    "pred_scores": pred_scores,
                    "pred_class_names": [class_names[lbl] for lbl in pred_labels],
                }

                results.append(result)

                print(f"\n📸 Image ID: {result['image_id']}")
                print(f"🟢 Predicted Boxes ({len(pred_boxes)}):")
                for box, label, score in zip(pred_boxes, result["pred_class_names"], pred_scores):
                    print(f"   {label} (Score: {score:.2f}): {box}")

                # 🔍 Plot predictions (optional)
                if plot:
                    plot_predictions(images[img_idx], pred_boxes, pred_labels, pred_scores, class_names, score_threshold)

            print(f"📦 Processed {i+1}/{len(dataloader_test)} batches...")

    # Save results
    with open(save_results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n✅ Test results saved to {save_results_path}")



def get_object_detection_model(num_classes):

    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
    
def get_transform(train):

    # train augmentation transforms
    if train:
        return A.Compose([
            A.HorizontalFlip(0.5),
            # ToTensorV2 converts image to pytorch tensor without div by 255
            ToTensorV2(p=1.0)
            ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

    else:
        return A.Compose([
            ToTensorV2(p=1.0)
            ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
        
        
# the function takes the original prediction and the iou threshold.
def apply_nms(orig_prediction, iou_thresh=0.3):

    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)

    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]

    return final_prediction

# function to convert a torchtensor back to PIL image
def torch_to_pil(img):
    return torchtrans.ToPILImage()(img).convert('RGB')