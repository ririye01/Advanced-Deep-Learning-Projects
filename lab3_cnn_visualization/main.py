import os
import sys
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from torchvision.models import vgg16, VGG16_Weights
from torchsummary import summary

# NOTE: TorchVision offers pre-trained weights for every provided architecture, using the PyTorch torch.hub. 
# Instancing a pre-trained model will download its weights to a cache directory. 
# This directory can be set using the TORCH_HOME environment variable
os.environ['TORCH_HOME'] = '/path/to/your/TORCH_HOME'

# VGG-16 using pretrained weights
vgg = vgg16(weights=VGG16_Weights.DEFAULT)

# Freeze VGG
vgg.eval()

# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg.to(device)

# Print the summary. NOTE: Feel free to experiment with different input sizes
# summary(vgg, (3, 32, 32))
# print(vgg)


# Using gradient techniques, find an input image that maximally excites the chosen filter.
target_conv_layer = 14 # "Conv2d-15" layer. The 15th layer in the network
filter_index = 2 # The 3rd filter in the "Conv2d-15"

# Define the loss function to maximize the activation of the chosen filter
loss_fn = torch.nn.MSELoss()


# Define a random input image or initialize with random noise
input_image = torch.randn(1, 3, 256, 256, requires_grad=True, device=device)  # Random noise image

# Define the optimizer # NOTE: The list of paramaters to optimize is specified. 
# Only the input image will get updated.
optimizer = torch.optim.Adam([input_image], lr=0.01)

# Number of optimization steps
num_steps = 100

# Optimization loop
for step in range(num_steps):
    # Forward pass. NOTE: This line selects the layers from the beginning of the VGG network (vgg.features) 
    # up to the layer specified by target_conv_layer, then feedforward "input_image". 
    output = vgg.features[: target_conv_layer](input_image)

    # Get the activations of the chosen filter
    activations = output[:, filter_index]
    
    # Compute mean squared error
    target_activations = torch.zeros_like(activations)
    loss = loss_fn(activations, target_activations)
    
    # Zero gradients, perform a backward pass, and update the input image
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print progress
    if (step+1) % 10 == 0:
        print(f'Step [{step+1}/{num_steps}], Loss: {loss.item()}')

# Convert the optimized image tensor to a numpy array
optimized_image = input_image.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)

# Clip values to stay within valid image range [0, 1]
optimized_image = np.clip(optimized_image, 0, 1)

# Convert numpy array to PIL image
optimized_image_pil = Image.fromarray((optimized_image * 255).astype(np.uint8))

# Save the image
optimized_image_pil.save("optimized_image.png")


