from torch import nn

# From https://en.wikipedia.org/wiki/AlexNet#/media/File:AlexNet_block_diagram.svg
class LeNet(nn.Module):
    def __init__(self, image_size: tuple[int, int], n_classes: int):
        super(LeNet, self).__init__()
        
        self.description = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5, 5), padding=2),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5)),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Flatten()
        )
        
        h, w = image_size
        
        self.classification = nn.Sequential(
            nn.Linear(in_features=..., out_features=120),
            nn.Tanh(),
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )
        
    def forward(self, x):
        descriptors = self.description(x)
        output = self.classification(descriptors)
        return output