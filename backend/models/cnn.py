import torch
import torch.nn as nn
import torch.nn.functional as F

class QuickDrawCNN(nn.Module):
    def __init__(self, num_classes=16):
        super(QuickDrawCNN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels = 1, #grayscale = 1 channel
            out_channels=32, #learn 32 different features
            kernel_size=3, #3x3 sliding window
            padding=1
        )

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout(p=0.25) #for regularization to prevent overfitting

        self.fc1 = nn.Linear(128*3*3, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):

        x = self.conv1(x) #applying convolution
        x = F.relu(x) #activation
        x = self.pool1(x) #downsample

        x = self.conv2(x) #applying convolution
        x = F.relu(x) #activation
        x = self.pool2(x) #downsample

        x = self.conv3(x) #applying convolution
        x = F.relu(x) #activation
        x = self.pool3(x) #downsample

        # Flatten from (batch, 128, 3, 3) to (batch, 1152)
        x = torch.flatten(x, 1)
        
        # Dropout
        x = self.dropout(x)
        
        # Fully connected
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    

if __name__ == '__main__':
    model = QuickDrawCNN(num_classes=16)
    
    print('Model architecture:')
    print(model)
    print(f'\nTotal trainable parameters: {model.count_parameters():,}')
    
    # Pass a fake batch through to verify shapes
    fake_batch = torch.randn(64, 1, 28, 28)
    output = model(fake_batch)
    
    print(f'\nInput shape:  {fake_batch.shape}')
    print(f'Output shape: {output.shape}')
    print('Shape check passed.' if output.shape == (64, 16) else 'ERROR: wrong output shape')
        





        