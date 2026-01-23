import torch.optim as optim
from trainers.basic_trainer import BasicTrainingModule

class WeightDecayTrainer(BasicTrainingModule):
    
    def __init__(self, model, loss_fn, lr=1e-3, weight_decay=1e-2, augmentations=None):
        super().__init__(model=model, loss_fn=loss_fn, lr=lr, augmentations=augmentations)
        self.weight_decay = weight_decay
        
    def configure_optimizers(self):
        return optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)