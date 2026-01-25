import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

class BasicTrainingModule(pl.LightningModule):
    """
    A minimal, reusable PyTorch Lightning module for training models.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to be trained.
    loss_fn : torch.nn.Module
        Loss function used for optimization
    lr : float, optional
        Learning rate for the optimizer. Default is 1e-5.
    optimizer_cls : type, optional
        Optimizer class to use. Default is torch.optim.Adam.
    """
    
    
    def __init__(self, 
                 model : nn.Module, 
                 loss_fn : nn.Module = nn.CrossEntropyLoss(), 
                 lr : float = 1e-5, 
                 optimizer_cls : optim.Optimizer = optim.AdamW,
                 augmentations = None
                 ):
        
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.optimizer_cls = optimizer_cls
        
        self.augmentation = augmentations
        
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        if self.augmentation is not None:
            x = self.augmentation(x)
        
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc",  acc, on_step=False, on_epoch=True)

        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True)
        
        return {"test_loss": loss}
        
    def configure_optimizers(self):
        
        optimizer = self.optimizer_cls(
            self.parameters(),
            lr=self.lr,
        )
        
        return optimizer
    
    