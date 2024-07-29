from module.shufflenetv2 import ShuffleNetV2
import torch
import torch.nn as nn

class LightClassifier(nn.Module):
    def __init__(self, classes, load_param, debug=False):
        super(LightClassifier, self).__init__()
        
        self.stage_repeats = [4, 8, 4]
        self.stage_out_channels = [-1, 24, 48, 96, 192]
        self.base = ShuffleNetV2(self.stage_repeats, self.stage_out_channels, load_param)

        # Shared convolutional layer
        self.shared_conv = nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(192)
        self.relu = nn.ReLU(inplace=True)

        # Add a global average pooling layer
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Add a fully connected layer for classification
        self.fc = nn.Linear(self.stage_out_channels[-1], classes)
        
        self.debug = debug
        
    def forward(self, x):
        if self.debug:
            print("forward ", x.size())
        _, _, P3 = self.base(x)
        if self.debug:
            print("base output ", P3.size())
        
        # Apply shared convolutional layer
        x = self.shared_conv(P3)
        x = self.bn(x)
        x = self.relu(x)
        
        if self.debug:
            print("after shared conv ", x.size())
        x = self.global_pool(x)
        if self.debug:
            print("after global pool ", x.size())
        features = x.view(x.size(0), -1)  # Flatten the tensor
        if self.debug:
            print("after tensor flattening ", features.size())
        logits = self.fc(features)
        if self.debug:
            print("final shape ", logits.size())
        return features, logits
    
if __name__ == '__main__':
    model = LightClassifier(2, False, True)
    test_data = torch.rand(1, 3, 200, 200)
    features, logits = model(test_data)
    print("final shape ", features.size())
    print("logits ", logits.size())