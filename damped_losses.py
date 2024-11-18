import torch
import numpy as np
from fastai.vision.all import *
from fastai import __version__
import torch
from torch.utils.data import Dataset, DataLoader
from efficientnet_pytorch import EfficientNet
from fastai.vision import *
from fastai import optimizer, losses, metrics
from functools import partial, wraps
import matplotlib.pyplot as plt
import random

# 0->Damped CE
class DampedCELoss(Module):
    def __init__(self, num_classes:int, delta:float=0.1,  axis:int = -1, reduction ='mean'):
        store_attr()


        
    def forward(self, pred, labels):
        def smoothstep_(x, delta):
            if delta == 0.0:
                return 1.0
            x = torch.clamp(x/delta, min=1e-6, max=1.0)
            return x * x * x * (x * (6.0 * x - 15.0) + 10.0)
        
        # CCE --> check reduction
        ce = F.cross_entropy(pred, labels, weight = None, reduction="none") 

        # damped Loss
        loss = smoothstep_(torch.exp(-ce),self.delta) * ce
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


class DampedCELossFlat(BaseLoss):
    "Same as `SCELoss`, but flattens input and target."
    y_int = True # y interpolation
    @use_kwargs_dict(keep=True, weight=None, ignore_index=-100, reduction='mean')
    def __init__(self, 
        *args, 
        num_classes,          
        delta = 0.1,
        axis = -1,
        **kwargs
    ): 
        super().__init__(DampedCELoss, *args, num_classes = num_classes, delta = delta, axis= axis,  **kwargs)
    
    def decodes(self, x:Tensor) -> Tensor:    
        "Converts model output to target format"
        return x.argmax(dim=self.axis)
    
    def activation(self, x:Tensor) -> Tensor: 
        "`nn.CrossEntropyLoss`'s fused activation function applied to model output"
        return F.softmax(x, dim=self.axis)


# 1-> Dump SCE Loss
class DumpSCELoss(Module):
    def __init__(self, num_classes:int,  alpha:float=0.2, beta:float=2.0, delta:float=1.0,  axis:int = -1, reduction ='mean'):
        store_attr()

    def forward(self, pred, labels):
        def smoothstep_(x, delta):
            if delta == 0.0:
                return 1.0
            x = torch.clamp(x/delta, min=1e-6, max=1.0)
            return x * x * x * (x * (6.0 * x - 15.0) + 10.0)
        
        # CCE --> check reduction
        ce = F.cross_entropy(pred, labels, weight = None, reduction="none") 

        # RCE
        pred = F.softmax(pred, dim=self.axis)
        pred = torch.clamp(pred, min=1e-3, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        l = self.alpha * ce + self.beta * rce.mean()
        loss = smoothstep_(l, self.delta) * l
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


class DumpledSCELossFlat(BaseLoss):
    "Same as `SCELoss`, but flattens input and target."
    y_int = True # y interpolation
    @use_kwargs_dict(keep=True, weight=None, ignore_index=-100, reduction='mean')
    def __init__(self, 
        *args, 
        num_classes,
        alpha = 0.2,
        beta = 2.0,         
        delta = 0.1,
        axis = -1,
        **kwargs
    ): 
        super().__init__(DumpSCELoss, *args, num_classes = num_classes, alpha=alpha, beta=beta, delta=delta, axis=axis, **kwargs)
    
    def decodes(self, x:Tensor) -> Tensor:    
        "Converts model output to target format"
        return x.argmax(dim=self.axis)
    
    def activation(self, x:Tensor) -> Tensor: 
        "`nn.CrossEntropyLoss`'s fused activation function applied to model output"
        return F.softmax(x, dim=self.axis)


    
    
    # 2-> RCE Loss Function 
class DumpRCELoss(Module):
    def __init__(self, num_classes:int, scale:float=1.0, delta:float=0.1, axis:int = -1, reduction:str='mean'):
         store_attr()

    def forward(self, pred, labels):
        def smoothstep_(x, delta):
            if delta == 0.0:
                return 1.0
            x = torch.clamp(x/delta, min=1e-6, max=1.0)
            return x * x * x * (x * (6.0 * x - 15.0) + 10.0)
        
        pred = F.softmax(pred, dim=self.axis)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        loss = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))
        loss = smoothstep_(loss, self.delta) * loss
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return self.scale * loss

    
class DumpedRCELossFlat(BaseLoss):
    """
    """
    y_int = True # y interpolation
    @use_kwargs_dict(keep=True, weight=None, reduction='mean')
    def __init__(self, 
        *args, 
        num_classes:int,
        scale:float=1.0, # Focusing parameter. Higher values down-weight easy examples' contribution to loss
        axis:int=-1, # Class axis
        delta = 0.1,
        **kwargs
    ):
        super().__init__(DumpRCELoss, *args, num_classes=num_classes, scale=scale, delta = delta, axis=axis, **kwargs)
        
    def decodes(self, x:Tensor) -> Tensor: 
        "Converts model output to target format"
        return x.argmax(dim=self.axis)
    
    def activation(self, x:Tensor) -> Tensor: 
        "`F.cross_entropy`'s fused activation function applied to model output"
        return F.softmax(x, dim=self.axis)  
    
 #5-> GCE Loss Function

class DumpedGCELoss(Module):
    def __init__(self, num_classes:int, q:float=0.7, delta:float = 0.1, reduction:str='mean'):
        store_attr()

    def forward(self, pred, labels):
        def smoothstep_(x, delta):
            if delta == 0.0:
                return 1.0
            x = torch.clamp(x/delta, min=1e-6, max=1.0)
            return x * x * x * (x * (6.0 * x - 15.0) + 10.0)
        
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        gce = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        loss = gce
        loss = smoothstep_(loss, self.delta) * loss
        if self.reduction == "mean":
            loss = loss.mean()
#             print("The Loss (Mean) is:", loss)
        elif self.reduction == "sum":
            loss = loss.sum()
#             print("The Loss (Sum) is", loss)
        return loss

    
    
class DumpedGCELossFlat(BaseLoss):
    """
    """
    y_int = True # y interpolation
    @use_kwargs_dict(keep=True, weight=None)
    def __init__(self, 
        *args, 
        num_classes:int,
        q:float=0.7, # Focusing parameter. Higher values down-weight easy examples' contribution to loss
        axis:int=-1, # Class axis
        delta = 0.1,
        **kwargs
    ):
        super().__init__(DumpedGCELoss, *args, num_classes=num_classes, q=q, delta = delta, axis=axis, **kwargs)
        
    def decodes(self, x:Tensor) -> Tensor: 
        "Converts model output to target format"
        return x.argmax(dim=self.axis)
    
    def activation(self, x:Tensor) -> Tensor: 
        "`F.cross_entropy`'s fused activation function applied to model output"
        return F.softmax(x, dim=self.axis)
    
# 7-> MAE Loss Function 
class DumpedMAELoss(Module):
    def __init__(self, num_classes:int, scale:float=1.0, delta:float=0.1, reduction:str='mean'):
        store_attr()

    def forward(self, pred, labels):
        def smoothstep_(x, delta):
            if delta == 0.0:
                return 1.0
            x = torch.clamp(x/delta, min=1e-6, max=1.0)
            return x * x * x * (x * (6.0 * x - 15.0) + 10.0)
        
        pred = F.softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        mae = 1. - torch.sum(label_one_hot * pred, dim=1)
        # Note: Reduced MAE
        # Original: torch.abs(pred - label_one_hot).sum(dim=1)
        # $MAE = \sum_{k=1}^{K} |\bm{p}(k|\bm{x}) - \bm{q}(k|\bm{x})|$
        # $MAE = \sum_{k=1}^{K}\bm{p}(k|\bm{x}) - p(y|\bm{x}) + (1 - p(y|\bm{x}))$
        # $MAE = 2 - 2p(y|\bm{x})$
        #
        loss = self.scale * mae
        loss = smoothstep_(loss, self.delta) * loss
        if self.reduction == "mean":
            loss = loss.mean()
#             print("The Loss (Mean) is:", loss)
        elif self.reduction == "sum":
            loss = loss.sum()
#             print("The Loss (Sum) is", loss)
        return loss
    
class DumpedMAELossFlat(BaseLoss):
    """
    """
    y_int = True # y interpolation
    @use_kwargs_dict(keep=True, weight=None)
    def __init__(self, 
        *args, 
        num_classes:int,
        scale:float=1.0, # Focusing parameter. Higher values down-weight easy examples' contribution to loss
        axis:int=-1, # Class axis
        delta:float = 0.1,
        **kwargs
    ):
        super().__init__(DumpedMAELoss, *args, num_classes=num_classes, scale=scale, axis=axis, **kwargs)
        
    def decodes(self, x:Tensor) -> Tensor: 
        "Converts model output to target format"
        return x.argmax(dim=self.axis)
    
    def activation(self, x:Tensor) -> Tensor: 
        "`F.cross_entropy`'s fused activation function applied to model output"
        return F.softmax(x, dim=self.axis)
   