import math
from abc import ABC
import torch


class MaskingSchedule(ABC): 
    def __init__(self, eps=1e-4):
        self.eps = eps

    def _alpha(self, t):
        raise NotImplementedError()

    def _dalpha(self, t):
        raise NotImplementedError()

    def alpha(self, t):
        return (1.0 - 2 * self.eps) * self._alpha(t) + self.eps

    def dalpha(self, t):
        return (1.0 - 2 * self.eps) * self._dalpha(t)

    def __call__(self, t):
        return self.alpha(t)
        
    def get_weight(self, t):
        return self.dalpha(t) / (1.0 - self.alpha(t))


class LinearSchedule(MaskingSchedule):        
    def _alpha(self, t):
        return 1.0 - t
    
    def _dalpha(self, t):
        return -1.0


class PolynomialSchedule(MaskingSchedule):
    def __init__(self, exponent, **kwargs):
        super().__init__(**kwargs)
        self.exponent = exponent

    def _alpha(self, t):
        return 1.0 - t ** self.exponent
    
    def _dalpha(self, t):
        return -self.exponent * t ** (self.exponent - 1.0)


class GeometricSchedule(MaskingSchedule):
    def __init__(self, beta_min, beta_max, **kwargs):
        super().__init__(**kwargs)
        self.beta_min = beta_min
        self.beta_max = beta_max

    def _alpha(self, t):
        return torch.exp(-self.beta_min ** (1.0 - t) * self.beta_max ** t)
    
    def _dalpha(self, t):
        inner = self.beta_min ** (1.0 - t) * self.beta_max ** t
        forward = torch.exp(-inner)
        return -forward * inner * math.log(self.beta_min / self.beta_max)


class CosineSchedule(MaskingSchedule):
    def _alpha(self, t):
        return 1.0 - torch.cos(torch.pi / 2.0 * (1.0 - t))
    
    def _dalpha(self, t):
        return -torch.pi / 2.0 * torch.sin(torch.pi / 2.0 * (1.0 - t))


def string_to_schedule(string, **kwargs):
    if string == "linear":
        return LinearSchedule(**kwargs)
    elif string == "cosine":
        return CosineSchedule(**kwargs)
    elif string == "geometric":
        return GeometricSchedule(**kwargs)
    elif string == "polynomial":
        return PolynomialSchedule(**kwargs)
    raise NotImplementedError()
