from typing import Callable
import thermalizer.models.misc as misc
import torch
from torch import nn
from torch.nn import functional as F
## Adapted from the beautiful repo at https://github.com/pdearena/pdearena/blob/main/pdearena/modules/twod_resnet.py

#######################################################################
#######################################################################
class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        activation: str = "relu",
        norm: bool = True,
        num_groups: int = 1,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True,padding_mode="circular")

        self.bn1 = nn.GroupNorm(num_groups, num_channels=planes) if norm else nn.Identity()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True,padding_mode="circular")
        self.bn2 = nn.GroupNorm(num_groups, num_channels=planes)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups, self.expansion * planes) if norm else nn.Identity(),
            )

        self.activation: nn.Module = misc.ACTIVATION_REGISTRY.get(activation, None)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # out = self.activation(self.bn1(self.conv1(x)))
        # out = self.bn2(self.conv2(out))
        out = self.conv1(self.activation(self.bn1(x)))
        out = self.conv2(self.activation(self.bn2(out)))
        out = out + self.shortcut(x)
        # out = self.activation(out)
        return out


class DilatedBasicBlock(nn.Module):
    """Basic block for Dilated ResNet

    Args:
        in_planes (int): number of input channels
        planes (int): number of output channels
        stride (int, optional): stride of the convolution. Defaults to 1.
        activation (str, optional): activation function. Defaults to "relu".
        norm (bool, optional): whether to use group normalization. Defaults to True.
        num_groups (int, optional): number of groups for group normalization. Defaults to 1.
    """

    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        activation: str = "relu",
        norm: bool = True,
        num_groups: int = 1,
    ):
        super().__init__()

        self.dilation = [1, 2, 4, 8, 4, 2, 1]
        dilation_layers = []
        for dil in self.dilation:
            dilation_layers.append(
                nn.Conv2d(
                    in_planes,
                    planes,
                    kernel_size=3,
                    stride=stride,
                    dilation=dil,
                    padding=dil,
                    bias=True,
                    padding_mode="circular"
                )
            )
        self.dilation_layers = nn.ModuleList(dilation_layers)
        self.norm_layers = nn.ModuleList(
            nn.GroupNorm(num_groups, num_channels=planes) if norm else nn.Identity() for dil in self.dilation
        )
        self.activation: nn.Module = misc.ACTIVATION_REGISTRY.get(activation, None)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer, norm in zip(self.dilation_layers, self.norm_layers):
            out = self.activation(layer(norm(out)))
        return out + x


class ResNet(nn.Module):
    """Class to support ResNet like feedforward architectures

    Config keys:
        n_input_scalar_components (int): Number of input scalar components in the model
        n_output_scalar_components (int): Number of output scalar components in the model
        block (Callable): BasicBlock or DilatedBasicBlock or FourierBasicBlock
        num_blocks (List[int]): Number of dilated blocks in each layer, and number of layers
        hidden_channels (int): Number of channels in the hidden layers
        activation (str): Activation function to use
        norm (bool): Whether to use normalization
    """

    #padding = 9

    def __init__(
        self,
        config,
    ):
        super().__init__()

        self.config=config
        self.config["model_type"]="DRN"
        self.n_input_scalar_components = self.config["input_channels"]
        self.n_output_scalar_components = self.config["output_channels"]
        self.in_planes = self.config["hidden_channels"]
        self.num_blocks = self.config["num_blocks"]
        insize = self.n_input_scalar_components
        self.conv_in1 = nn.Conv2d(
            insize,
            self.in_planes,
            kernel_size=1,
            bias=True,
        )
        self.conv_in2 = nn.Conv2d(
            self.in_planes,
            self.in_planes,
            kernel_size=1,
            bias=True,
        )
        self.conv_out1 = nn.Conv2d(
            self.in_planes,
            self.in_planes,
            kernel_size=1,
            bias=True,
        )
        self.conv_out2 = nn.Conv2d(
            self.in_planes,
            insize,
            kernel_size=1,
            bias=True,
        )

        self.layers = nn.ModuleList(
            [
                self._make_layer(
                    DilatedBasicBlock,
                    self.in_planes,
                    self.num_blocks[i],
                    stride=1,
                    activation=self.config["activation"],
                    norm=self.config["norm"],
                )
                for i in range(len(self.num_blocks))
            ]
        )
        self.activation: nn.Module = misc.ACTIVATION_REGISTRY.get(self.config["activation"], None)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")

    def _make_layer(
        self,
        block: Callable,
        planes: int,
        num_blocks: int,
        stride: int,
        activation: str,
        norm: bool = True,
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    stride,
                    activation=activation,
                    norm=norm,
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def __repr__(self):
        return "ResNet"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        #x = x.reshape(x.size(0), -1, *x.shape[3:])  # collapse T,C
        
        # prev = x.float()
        x = self.activation(self.conv_in1(x.float()))
        x = self.activation(self.conv_in2(x.float()))

        #if self.padding > 0:
        #    x = F.pad(x, [0, self.padding, 0, self.padding])

        for layer in self.layers:
            x = layer(x)

        #if self.padding > 0:
        #    x = x[..., : -self.padding, : -self.padding]

        x = self.activation(self.conv_out1(x))
        x = self.conv_out2(x)

        return x