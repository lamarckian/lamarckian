"""
Copyright (C) 2020

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
import torch
import torch.nn as nn


class Channel(object):
    def __init__(self, channel):
        self.channel = channel

    def __call__(self):
        return self.channel

    def set(self, channel):
        self.channel = channel

    def next(self, channel):
        self.channel = channel
        return self.channel

from . import fc


def normc_initializer(std=1.0):
    def initializer(tensor):
        tensor.data.normal_(0, 1)
        tensor.data *= std / torch.sqrt(
            tensor.data.pow(2).sum(1, keepdim=True))

    return initializer


def same_padding(in_size, filter_size, stride_size):
    """Note: Padding is added to match TF conv2d `same` padding. See
    www.tensorflow.org/versions/r0.12/api_docs/python/nn/convolution

    Args:
        in_size (tuple): Rows (Height), Column (Width) for input
        stride_size (Union[int,Tuple[int, int]]): Rows (Height), column (Width)
            for stride. If int, height == width.
        filter_size (tuple): Rows (Height), column (Width) for filter

    Returns:
        padding (tuple): For input into torch.nn.ZeroPad2d.
        output (tuple): Output shape after padding and convolution.
    """
    in_height, in_width = in_size
    if isinstance(filter_size, int):
        filter_height, filter_width = filter_size, filter_size
    else:
        filter_height, filter_width = filter_size
    stride_height, stride_width = stride_size

    out_height = np.ceil(float(in_height) / float(stride_height))
    out_width = np.ceil(float(in_width) / float(stride_width))

    pad_along_height = int(
        ((out_height - 1) * stride_height + filter_height - in_height))
    pad_along_width = int(
        ((out_width - 1) * stride_width + filter_width - in_width))
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    output = (out_height, out_width)
    return padding, output


def get_activation_fn(name, framework="tf"):
    """Returns a framework specific activation function, given a name string.

    Args:
        name (str): One of "relu" (default), "tanh", "swish", or "linear".
        framework (str): One of "tf" or "torch".

    Returns:
        A framework-specific activtion function. e.g. tf.nn.tanh or
            torch.nn.ReLU. None if name in ["linear", None].

    Raises:
        ValueError: If name is an unknown activation function.
    """
    if framework == "torch":
        if name in ["linear", None]:
            return None
        if name == "swish":
            from ray.rllib.utils.torch_ops import Swish
            return Swish
        if name == "relu":
            return nn.ReLU
        elif name == "tanh":
            return nn.Tanh
    else:
        if name in ["linear", None]:
            return None
        tf1, tf, tfv = try_import_tf()
        fn = getattr(tf.nn, name, None)
        if fn is not None:
            return fn

    raise ValueError("Unknown activation ({}) for framework={}!".format(
        name, framework))


class SlimConv2d(nn.Module):
    """Simple mock of tf.slim Conv2d"""

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel,
            stride,
            padding,
            # Defaulting these to nn.[..] will break soft torch import.
            initializer="default",
            activation_fn="default",
            bias_init=0):
        super(SlimConv2d, self).__init__()
        layers = []
        # Padding layer.
        if padding:
            layers.append(nn.ZeroPad2d(padding))
        # Actual Conv2D layer (including correct initialization logic).
        conv = nn.Conv2d(in_channels, out_channels, kernel, stride)
        if initializer:
            if initializer == "default":
                initializer = nn.init.xavier_uniform_
            initializer(conv.weight)
        nn.init.constant_(conv.bias, bias_init)
        layers.append(conv)
        # Activation function (if any; default=ReLu).
        if isinstance(activation_fn, str):
            if activation_fn == "default":
                activation_fn = nn.ReLU
            else:
                activation_fn = get_activation_fn(activation_fn, "torch")
        if activation_fn is not None:
            layers.append(activation_fn())
        # Put everything in sequence.
        self._model = nn.Sequential(*layers)

    def forward(self, x):
        return self._model(x)


class SlimFC(nn.Module):
    """Simple PyTorch version of `linear` function"""

    def __init__(self,
                 in_size,
                 out_size,
                 initializer=None,
                 activation_fn=None,
                 use_bias=True,
                 bias_init=0.0):
        super(SlimFC, self).__init__()
        layers = []
        # Actual Conv2D layer (including correct initialization logic).
        linear = nn.Linear(in_size, out_size, bias=use_bias)
        if initializer:
            initializer(linear.weight)
        if use_bias is True:
            nn.init.constant_(linear.bias, bias_init)
        layers.append(linear)
        # Activation function (if any; default=None (linear)).
        if isinstance(activation_fn, str):
            activation_fn = get_activation_fn(activation_fn, "torch")
        if activation_fn is not None:
            layers.append(activation_fn())
        # Put everything in sequence.
        self._model = nn.Sequential(*layers)

    def forward(self, x):
        return self._model(x)


class ActorCriticModel(nn.Module):
    def __init__(self, inputs, outputs, **kwargs):
        """Residual deep net + fully connected wide net."""
        super().__init__()
        image, in_image_size = inputs[1]['shape'][0], inputs[1]['shape'][1:]
        (in_vector_size,), in_ch, out_size, n_outputs = inputs[0]['shape'], image, in_image_size, outputs['discrete']

        # IMPALA architecture conv part
        conv_layers = list()
        for (out_ch, num_blocks) in [(16, 2), (32, 2), (32, 2), (32, 2)]:
            # Downscale
            padding, out_size = same_padding(out_size, filter_size=3, stride_size=[1, 1])
            conv_layers.append(SlimConv2d(in_ch, out_ch, kernel=3, stride=1, padding=padding, activation_fn=None))

            padding, out_size = same_padding(out_size, filter_size=3, stride_size=[2, 2])
            conv_layers.append(nn.ZeroPad2d(padding))
            conv_layers.append(nn.MaxPool2d(kernel_size=3, stride=2))

            # Residual blocks
            for _ in range(num_blocks):
                res = ResidualBlock(i_channel=out_ch, o_channel=out_ch, in_size=out_size)
                conv_layers.append(res)

            padding, out_size = res.padding, res.out_size
            in_ch = out_ch

        conv_layers.append(nn.ReLU(inplace=True))
        self._conv_net = nn.Sequential(*conv_layers)

        # FC net
        fc_layers = list()
        conv_flatten_size = self._conv_net(torch.empty((1, image, *in_image_size))).flatten().size()[0]  # 960
        prev_layer_size = conv_flatten_size + in_vector_size  # we concat conv_out and vector_in as FC's input
        for hidden in [256, 256]:
            fc_layers.append(SlimFC(in_size=prev_layer_size, out_size=hidden,
                                    initializer=normc_initializer(1.0), activation_fn='relu'))
            prev_layer_size = hidden

        self._fc_net = nn.Sequential(*fc_layers)

        # build the logits and value heads
        self._logits = SlimFC(in_size=prev_layer_size, out_size=n_outputs,
                              initializer=normc_initializer(0.01), activation_fn=None)
        self._value = SlimFC(in_size=prev_layer_size, out_size=1,
                             initializer=normc_initializer(0.01), activation_fn=None)

    def forward(self, x_1d, x_3d, get_value=True):
        x_3d = x_3d.float()
        x_3d = self._conv_net(x_3d)
        x_3d = torch.flatten(x_3d, start_dim=1)
        x = torch.cat((x_3d, x_1d), dim=-1)
        x = self._fc_net(x)
        logits = self._logits(x)
        if get_value:
            return logits, self._value(x)
        else:
            return logits


class ResidualBlock(nn.Module):
    def __init__(self, i_channel, o_channel, in_size=(72, 96), kernel_size=3, stride=1):
        """Implement a two-layer residual block."""
        super().__init__()
        self._relu = nn.ReLU(inplace=True)

        padding, out_size = same_padding(in_size, kernel_size, [stride, stride])
        self._conv1 = SlimConv2d(i_channel, o_channel, kernel=3, stride=stride, padding=padding, activation_fn=None)

        padding, out_size = same_padding(out_size, kernel_size, [stride, stride])
        self._conv2 = SlimConv2d(o_channel, o_channel, kernel=3, stride=stride, padding=padding, activation_fn=None)

        self.padding, self.out_size = padding, out_size

    def forward(self, x):
        out = self._relu(x)
        out = self._conv1(out)
        out = self._relu(out)
        out = self._conv2(out)
        out += x
        return out
