import torch.nn as nn
from typing import *

class BaseModel(nn.Module):
    """
    Generic linear model
    """
    
    def __init__(self, widths: List[int]):
        """
        Only the layer widths are expected. For example [3 * 224 * 224, 32, 1], would create\n
        a model that has 1 layer (3 * 224 * 224 -> 32) as descriptor head and another as\n
        classification head (32 -> 1).
        
        The last width given will be the one for the classification head.
        """

        super(BaseModel, self).__init__()
        
        descriptor_modules = [nn.Flatten()]
        relu = nn.ReLU()
        layer_number = len(widths) - 1
        current_features = widths[0]
        
        for idx, out_features in enumerate(widths[1:-1]):
            descriptor_modules.append(nn.Linear(in_features=current_features, out_features=out_features))
            descriptor_modules.append(relu)
            current_features = out_features
            

        self.descriptor_head = nn.Sequential(*descriptor_modules)
        self.classification_head = nn.Linear(in_features=current_features, out_features=widths[-1])


    def forward(self, x):
        
        x = self.descriptor_head(x)
        x = self.classification_head(x)

        return x
    
    def get_descriptors(self, x):
        
        return self.descriptor_head(x)