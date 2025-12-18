from torch import nn

# from https://en.wikipedia.org/wiki/AlexNet#/media/File:AlexNet_block_diagram.svg
class AlexNet(nn.Module):
    def __init__(self, image_size: tuple[int, int], n_classes: int):
        super(AlexNet, self).__init__()
        self.description = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11, 11), stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.Flatten()
        )
        
        h, w = image_size
        h_out = (((((((h - 11) // 4 + 1) - 3) // 2 + 1) - 3) // 2 + 1) - 3) // 2 + 1
        w_out = (((((((w - 11) // 4 + 1) - 3) // 2 + 1) - 3) // 2 + 1) - 3) // 2 + 1

        self.classification = nn.Sequential(
            nn.Linear(in_features=256 * h_out * w_out, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=n_classes),
            # No softmax because CrossEntropyLoss already includes it
        )
        
    def forward(self, x):
        descriptors = self.description(x)
        output = self.classification(descriptors)
        return output
        