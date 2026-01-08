import torch

from abc import ABC


class MaskingSchedule(ABC):    
    def __call__(self, t):
        raise NotImplementedError()
        
    def get_weight(self, t):
        raise NotImplementedError()
        

class LinearSchedule(MaskingSchedule):        
    def __call__(self, t):
        return 1 - t
    
    def get_weight(self, t):
        return -1 / t


class PolynomialSchedule(MaskingSchedule):
    def __init__(self, exponent):
        self.exponent = exponent

    def __call__(self, t):
        return 1 - t ** self.exponent
    
    def get_weight(self, t):
        return -self.exponent / t

class GeometricSchedule(MaskingSchedule):
    def __init__(self, beta_min, beta_max):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def __call__(self, t):
        return torch.exp(-self.beta_min ** (1 - t) * self.beta_max ** t)
    
    def get_weight(self, t):
        inner = self.beta_min ** (1 - t) * self.beta_max ** t
        forward = torch.exp(-inner)
        return -forward / (1 - forward) * inner * torch.log(self.beta_min / self.beta_max)  # the paper might have accidentally put sigma_min instead of beta_min I think

class CosineSchedule(MaskingSchedule):
    def __call__(self, t):
        return 1 - torch.cos(torch.pi / 2 * (1 - t))
    
    def get_weight(self, t):
        return -torch.pi / 2 * torch.tan(torch.pi / 2 * (1 - t))
