import itertools
import torch
import torch.nn as nn
from torch.autograd.function import Function


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class FrozenContext:
    def __init__(self, *params):
        self.params = list(itertools.chain(*params))

    def __enter__(self):
        for p in self.params:
            p.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for p in self.params:
            p.requires_grad = True


class GradientReversalLayer(nn.Module):
    """
    Module implementing a Gradient Reversal Layer.
    Input: x, lambda
    Output: x

    It remembers lambda and the gradient is multiplied by -lambda when backpropagating
    """

    class _GRL(Function):

        @staticmethod
        def forward(ctx, x, lambda_val):
            ctx.lambda_val = lambda_val

            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad_out):
            grad_in = grad_out.neg() * ctx.lambda_val

            return grad_in, None

    def __init__(self):
        self.grl = GradientReversalLayer._GRL()
        super().__init__()

    def forward(self, x: torch.Tensor, lambda_val=0):
        # If training lambda_val must not be None
        # assert (lambda_val is None and not self.training) or (lambda_val is not None and self.training)
        return self.grl.apply(x, lambda_val)


class Identity(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class PluggableSequential(nn.Sequential):

    def __init__(self, *args, plug_position: int = 0, **kwargs):
        """
        Like Sequential but you can plug a secondary branch
        :param args: Submodules
        :param plug_position: Secondary branch position.
        0 1 2 3 4 5 6
         M M M M M M
        :param kwargs:
        """
        super().__init__(*args, **kwargs)

        self.plug_position = plug_position

    def forward(self, x):
        assert isinstance(self.plug_position, int)
        assert 0 <= self.plug_position <= len(self)

        plug_output = None
        for i, module in enumerate(self):
            if i == self.plug_position:
                plug_output = x
            x = module(x)

        if self.plug_position == len(self):
            plug_output = x

        return x, plug_output
