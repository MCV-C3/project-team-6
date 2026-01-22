import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from experiments.squeeze_excitation_experiment import create_model_for_experiment

class DistillationTrainingModule(pl.LightningModule):
    
    def __init__(self, 
                 student: nn.Module, 
                 lr: float = 1e-5, 
                 temperature: float = 4.0, 
                 alpha: float = 0.5,
                 optimizer_cls : optim.Optimizer = optim.AdamW,
                 augmentations = None):

        super().__init__()
        self.student = student
        
        checkpoint_path = '/ghome/group06/workspace/project-team-6/Week4/checkpoints/Definitive model/best_test_accuracy.pt'
        teacher = create_model_for_experiment("Sweep", num_classes=8)
        checkpoint = torch.load(checkpoint_path)
        teacher.load_state_dict(checkpoint['model_state_dict'])
        
        self.teacher = teacher
        self.teacher.eval()  # freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        self.lr = lr
        self.temperature = temperature
        self.alpha = alpha
        
        self.ce_loss = nn.CrossEntropyLoss()  # hard target loss
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')  # soft target loss
        
        self.optimizer_cls = optimizer_cls
        
        self.augmentation = augmentations
    
    def forward(self, x):
        return self.student(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        if self.augmentation is not None:
            x = self.augmentation(x)
        
        student_logits = self.student(x)
        with torch.no_grad():
            teacher_logits = self.teacher(x)
        
        # Hard target loss
        hard_loss = self.ce_loss(student_logits, y)
        
        # Soft target loss (KL divergence between softened logits)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = self.kl_loss(student_log_probs, teacher_probs) * (self.temperature ** 2)
        
        # Combined loss
        loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        preds = torch.argmax(student_logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc",  acc, on_step=False, on_epoch=True)
        
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.student(x)
        loss = self.ce_loss(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True)
        
        return {"test_loss": loss}
    
    def configure_optimizers(self):
        
        optimizer = self.optimizer_cls(
            self.student.parameters(),
            lr=self.lr,
        )
        
        return optimizer
