import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import shutil
from module.shufflenetv2 import ShuffleNetV2
import torchvision.transforms.functional as F


class LightClassifier(nn.Module):
    def __init__(self, classes, load_param, debug=False):
        super(LightClassifier, self).__init__()
        
        self.stage_repeats = [4, 8, 4]
        self.stage_out_channels = [-1, 24, 48, 96, 192]
        self.base = ShuffleNetV2(self.stage_repeats, self.stage_out_channels, load_param)

        # Add a global average pooling layer
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Add a fully connected layer for classification
        self.fc = nn.Linear(self.stage_out_channels[-1], classes)
        
        self.debug = debug
        
    def forward(self, x):
        if self.debug :
            print("forward ", x.size())
        _, _, P3 = self.base(x)
        if self.debug :
            print("base output ", P3.size())
        x = self.global_pool(P3)
        if self.debug :
            print("after global pool ", x.size())
        features = x.view(x.size(0), -1)  # Flatten the tensor
        if self.debug :
            print("after tensor flattening ", x.size())
        logits = self.fc(features)
        if self.debug :
            print("final shape ", x.size())
        return features, logits

def load_model(model_path, device):
    model = LightClassifier(classes=2, load_param=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

import cv2
import numpy as np

def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((192, 192)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    # image_cp = F.adjust_saturation(image, 2.6)
    image = preprocess(image)
    
    # image = image[[2, 1, 0], :, :]  # Reorder the channels
    
    
    # cv2.imshow("sat " , np.array(image_cp))
    # cv2.waitKey(0)
    # image = image 

    return image.unsqueeze(0)  # Add batch dimension

def run_inference(model, image_folder, device):
    classes = ['other', 'red']
    image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.endswith(('jpg', 'jpeg', 'png'))]
    
    # Create directories for classified images
    output_folders = {cls: os.path.join("/home/achintya-trn0175/Downloads/trafficLightClassification/custom_video_inference_700_85_15_ncnn", cls) for cls in classes}
    for folder in output_folders.values():
        if not os.path.exists(folder):
         os.makedirs(folder, exist_ok=True)
    
    for image_path in image_paths:
        image = preprocess_image(image_path).to(device)
        with torch.no_grad():
            _, logits = model(image)
            prediction = torch.argmax(logits, dim=1).item()
            predicted_class = classes[prediction]
            print(f'Image: {image_path}, Prediction: {predicted_class}')

            # Copy the image to the respective folder
            #print("saved path ",  output_folders[predicted_class])
            shutil.copy(image_path, output_folders[predicted_class])

if __name__ == '__main__':
    # Paths
    model_path = '/home/achintya-trn0175/Downloads/trafficLightClassification/weights/reduced_size/tflt_weight_loss:0.760505_190-epoch.pth'
    image_folder = '/home/achintya-trn0175/Downloads/test_traffic'

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = load_model(model_path, device)

    # Run inference
    run_inference(model, image_folder, device)

