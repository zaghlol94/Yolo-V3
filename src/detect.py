import argparse
import config
import torch
import torch.optim as optim
import torchvision.transforms as T
from model import YOLOv3
from PIL import Image
from utils import (
    cells_to_bboxes,
    load_checkpoint,
    non_max_suppression,
    plot_image
)

parser = argparse.ArgumentParser(description="object detection based on pascal-voc objects")
parser.add_argument("-I", "--image_path", type=str, required=True, help="path of the image to recognise objects in it")
args = parser.parse_args()
transform = T.Compose([T.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)), T.ToTensor()])

model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
optimizer = optim.Adam(
    model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
)

load_checkpoint(
    config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
)

scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
).to(config.DEVICE)

image = Image.open(args.image_path).convert("RGB")
image = transform(image).unsqueeze(0)
model.eval()

image = image.to("cuda")
with torch.no_grad():
    out = model(image)
    bboxes = [[] for _ in range(image.shape[0])]
    for i in range(3):
        batch_size, A, S, _, _ = out[i].shape
        anchor = scaled_anchors[i]
        boxes_scale_i = cells_to_bboxes(
            out[i], anchor, S=S, is_preds=True
        )
        for idx, (box) in enumerate(boxes_scale_i):
            bboxes[idx] += box
    for i in range(batch_size):
        nms_boxes = non_max_suppression(
            bboxes[i], iou_threshold=0.6, threshold=0.5, box_format="midpoint",
        )
        plot_image(image[i].permute(1, 2, 0).detach().cpu(), nms_boxes)
