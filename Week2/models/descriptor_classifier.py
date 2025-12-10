from torch import nn

class DescriptorClassifier(nn.Module):
    def __init__(self, descriptor_head: nn.Module, classification_head: nn.Module):
        super(DescriptorClassifier, self).__init__()

        self.descriptor_head = descriptor_head
        self.classification_head = classification_head

    def forward(self, x):
        descriptors = self.descriptor_head(x)
        output = self.classification_head(descriptors)
        return output

    def get_descriptors(self, x):
        return self.descriptor_head(x)


def make_like_simple(input_d: int, hidden_d: int, output_d: int) -> DescriptorClassifier:
    # Imitate the model given by the teachers
    descriptor_head = nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_d, hidden_d),
        nn.ReLU(),
        nn.Linear(hidden_d, hidden_d),
        nn.ReLU(),
    )
    classification_head = nn.Linear(hidden_d, output_d)
    return DescriptorClassifier(descriptor_head, classification_head)
