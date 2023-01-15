import torch
import ptens as p
class _Dropout(torch.nn.Module):
    def __init__(self, prob: float = 0.5, inplace: bool = false) -> None:
        super().__init__()
        if prob<0 or prob>1:
            raise ValueError("Dropout probability has to be between 0 and 1, "
                             "but got {}".format(prob))
        self.prob = prob
        self.inplace = inplace
        
    def extra_repr(self) -> str:
        return 'prob={}, inplace={}'.format(self.prob, self.inplace)

#class Dropout(_Dropout):
#    def forward(self, x: p.ptensor) -> p.ptensor:
        r"""
        Randomly zeroes some of the atoms of the input ptensor with probability `p`
        using samples from a Bernoulli distribution. Each channel will be zeroed out
        independently on every forward call.
        """
#        return torch.nn.functional.dropout(x, self.prob, self.training, self.inplace)

class Dropout_0P(_Dropout):
    def forward(self, x: p.ptensors0) -> p.ptensors0:
        r"""
        Randomly zero out entire channels(a channel is a 1D feature map)
        Each channel will be zeroed out independently on every forward call
        with probability `p` using samples from a Bernoulli distribution. 
        """
        return torch.nn.functional.dropout1d(x, self.prob, self.training, self.inplace)

class Dropout_1P(_Dropout):
    def forward(self, x: p.ptensors1) -> p.ptensors1:
        r"""
        Randomly zero out entire channels(a channel is a 2D feature map)
        Each channel will be zeroed out independently on every forward call
        with probability `p` using samples from a Bernoulli distribution. 
        """
        return torch.nn.functional.dropout2d(x, self.prob, self.training, self.inplace)

class Dropout_2P(_Dropout):
    def forward(self, x: p.ptensors2) -> p.ptensors2:
        r"""
        Randomly zero out entire channels(a channel is a 3D feature map)
        Each channel will be zeroed out independently on every forward call
        with probability `p` using samples from a Bernoulli distribution.
        """
        return torch.nn.functional.dropout3d(x, self.prob, self.training, self.inplace)

class AlphaDropout_0P(_Dropout):
    def forward(self, x: p.ptensors0) -> p.ptensors0:
        return torch.nn.functional.alpha_dropout(x, self.prob, self.training)  
class AlphaDropout_1P(_Dropout):
    def forward(self, x: p.ptensors1) -> p.ptensors1:
        return torch.nn.functional.alpha_dropout(x, self.prob, self.training)
class AlphaDropout_2P(_Dropout):
    def forward(self, x: p.ptensors2) -> p.ptensors2:
        return torch.nn.functional.alpha_dropout(x, self.prob, self.training)

class FeatureAlphaDropout_0P(_Dropout):
    def forward(self, x: p.ptensors0) -> p.ptensors0:
        return torch.nn.functional.feature_alpha_dropout(x, self.prob, self.training)
class FeatureAlphaDropout_1P(_Dropout):
    def forward(self, x: p.ptensors1) -> p.ptensors1:
        return torch.nn.functional.feature_alpha_dropout(x, self.prob, self.training)
class FeatureAlphaDropout_2P(_Dropout):
    def forward(self, x: p.ptensors2) -> p.ptensors2:
        return torch.nn.functional.feature_alpha_dropout(x, self.prob, self.training)
