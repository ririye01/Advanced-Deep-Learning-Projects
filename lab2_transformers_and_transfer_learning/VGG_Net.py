######################################################
# CODE INSPIRED BY THE FOLLOWING VIDEO:              #
# https://www.youtube.com/watch?v=ACmuBbuXn20&t=12s  #
######################################################

import torch
import torch.nn as nn

from typing import List, Dict, Union, Any


# Layer prefixes for VGG11 – VGG19
VGG_TYPES: Dict[str, List[Any]] = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


class VGG_Net(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 1000,
        vgg_type: str = "VGG16",
    ) -> None:
        """
        This implementation of VGG_19 works on 64 x 64 x `num_channels` images.

        Args:
            input_channels (int): Color channels for image. Default is 3.
            num_classes (int): Number of classes to predict. Default is 1000 for ImageNet.
            vgg_type (str): Specified VGG type for user. Default is "VGG16".
        """
        super(VGG_Net, self).__init__()

        # Private variables
        self.input_channels = input_channels
        self.num_classes = num_classes

        # Specify convolutional layers based on `vgg_type` chosen by user
        self.convolutional_layers: nn.Sequential = self._create_convolutional_layers(VGG_TYPES[vgg_type])

        # Establish the forward connecting sequence
        output_channel_count = 512
        self.fcs = nn.Sequential(
            nn.Linear(output_channel_count * 2 * 2, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )


    def _create_convolutional_layers(self, vgg_architecture: List[Union[int, str]]) -> nn.Sequential:
        # Declare neural network layers as empty to begin with, then append Convolutional Blocks
        # and update input channels as we forward propogate through the network.
        neural_network_layers: List[Union[nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d]] = []
        input_channel_count = self.input_channels

        for output_channel_label in vgg_architecture:
            # Add a convolutional layer if we have an integer output channel label
            if type(output_channel_label) == int:
                output_channel_count = output_channel_label

                # Declare layers of the Conv2D Block
                conv2d_layer = nn.Conv2d(
                    in_channels=input_channel_count,
                    out_channels=output_channel_count,
                    kernel_size=(3,3),
                    stride=(1,1),
                    padding=(1,1),
                )
                batch_normalization_layer = nn.BatchNorm2d(output_channel_count)
                relu = nn.ReLU()

                # Append labels
                neural_network_layers.append(conv2d_layer)
                neural_network_layers.append(batch_normalization_layer)
                neural_network_layers.append(relu)

                # Update input channels to be former output channel count
                input_channel_count = output_channel_count

            # Add a max pooling layer to reduce noise if we approach a "M"
            elif output_channel_label == "M":
                max_pooling_layer = nn.MaxPool2d(
                    kernel_size=(2,2),
                    stride=(2,2),
                )
                neural_network_layers.append(max_pooling_layer)

        return nn.Sequential(*neural_network_layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convolutional_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x
