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
                 loss_fn : nn.Module = nn.CrossEntropyLoss(), 
                 lr : float = 1e-5, 
                 optimizer_cls : optim.Optimizer = optim.AdamW,
                 num_classes : int = 8
                 ):
        
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.optimizer_cls = optimizer_cls
        
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_loss = torchmetrics.MeanMetric()
        self.train_loss = torchmetrics.MeanMetric()
        
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        self.train_loss.update(loss)
        self.train_acc.update(y_hat, y)

        return {"loss": loss}
    
    def on_train_epoch_end(self, outputs):
        self.log("train_loss", self.train_loss.compute(), prog_bar=True)
        self.log("train_acc", self.train_acc.compute(), prog_bar=True)
        self.train_loss.reset()
        self.train_acc.reset()
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.val_loss.update(loss)
        self.val_acc.update(y_hat, y)
        return {"test_loss": loss}
        

    def on_validation_epoch_end(self):
        self.log("val_loss", self.val_loss.compute(), prog_bar=True)
        self.log("val_acc", self.val_acc.compute(), prog_bar=True)
        self.val_loss.reset()
        self.val_acc.reset()
        
    def configure_optimizers(self):
        
        optimizer = self.optimizer_cls(
            self.parameters(),
            lr=self.lr,
        )
        
        return optimizer