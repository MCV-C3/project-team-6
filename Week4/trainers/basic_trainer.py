import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torchmetrics

class BasicTrainingModule(pl.LightningModule):
    """
    A minimal, reusable PyTorch Lightning module for training models.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to be trained.
    loss_fn : torch.nn.Module
        Loss function used for optimization.
    lr : float, optional
        Learning rate for the optimizer. Default is 1e-5.
    optimizer_cls : type, optional
        Optimizer class to use. Default is torch.optim.Adam.
    """
    
    
    def __init__(self, 
                 model : nn.Module, 
                 loss_fn : nn.Module = nn.CrossEntropyLoss, 
                 lr : float = 1e-5, 
                 optimizer_cls : optim.Optimizer = optim.AdamW,
                 num_classes : int = 8
                 ):
        
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.optimizer_cls = optimizer_cls
        
        self.train_acc = torchmetrics.Accuracy(num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(num_classes=num_classes)
        
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        self.train_acc.update(y_hat, y)

        return {"loss": loss}
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        acc = self.train_acc.compute()
        self.log("train_loss", avg_loss)
        self.log("train_acc", acc)
        self.train_acc.reset()
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        self.val_acc.update(y_hat, y)
        return {"test_loss": loss}
        
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        acc = self.val_acc.compute()
        self.log("test_loss", avg_loss)
        self.log("test_acc", acc)
        self.val_acc.reset()
        
    def configure_optimizers(self):
        
        optimizer = self.optimizer_cls(
            self.parameters(),
            lr=self.lr,
        )
        
        return optimizer