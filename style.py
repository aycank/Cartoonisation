# Import
from PIL import Image
from io import BytesIO
from torchvision import transforms, models
import torch
import torch.optim as optim
import numpy as np
import requests
import time


# Load image, making sure the image size is <= 512 pixels in the x & y dimension
def load_image(img_path):
    image_size = 512  # Max image size
    mean = (0.485, 0.456, 0.406)  # Mean and std of ImageNet
    std = (0.229, 0.224, 0.225)
    if "https" in img_path:  # Check whether the image is from a URL
        response = requests.get(img_path)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(img_path)
    if max(image.size) > image_size:  # If the image that is loaded has a larger size, max image size input given
        size = image_size
    else:
        size = max(image.size)

    image_transform = transforms.Compose(
        [
            transforms.Resize((size, size)),   # Transform images to the same size in order to properly compute loss
            transforms.ToTensor(),  # Transform to Tensor for faster computation times
            transforms.Normalize(mean, std)  # Normalize(mean, standard deviation)
        ]
    )
    image = image_transform(image).unsqueeze(0)  # Squeezing it in to 1 Array - add additional dimension for batch size
    return image


# Function for Un-Normalising an image
# Converting it from a Tensor image to a Numpy image
def image_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = (image * np.array((0.229, 0.224, 0.225))) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image


def get_features(image, model):
    # Layers are matching the Gatys' paper (2016)
    # The numbers 0, 5, 10, 19, 21, 28 are all after MaxPool2D
    layers = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1', '19': 'conv4_1', '21': 'conv4_2', '28': 'conv5_1'}
    features = {}
    x = image
    for number, layer in model._modules.items():  # _modules.items() unpacks the vgg layers
        x = layer(x)
        if number in layers:
            features[layers[number]] = x
    return features


# Function to calculate the Gram Matrix of a given Tensor
def gram_matrix(tensor):
    _, depth, width, height = tensor.size()  # Get the batch_size, depth, width, and height of the Tensor
    tensor = tensor.view(depth, height*width)  # Reshape so it's multiplying the features for each channel
    gram = torch.mm(tensor, tensor.t())  # Calculate the Gram Matrix
    return gram


def stylize(content_image, style_image, total_steps, learning_rate, alpha, beta):
    start = time.time()
    '''
    Get the VGG19 "features". The model is pretrained.
    Freeze all VGG layer parameters - only optimising the target image as no need to train the model
    '''
    vgg = models.vgg19(pretrained=True).features
    for param in vgg.parameters():
        param.requires_grad_(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg.to(device).eval()

    show_every = 100  # Print the progress every number of steps
    content = load_image(content_image).to(device)
    style = load_image(style_image).to(device)  # Resize style to match the content

    # Get Content and Style features only once before training
    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)

    # Calculate the Gram Matrices for each layer of the chosen style representation
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    # Weights for each style layer
    style_weight = {'conv1_1': 1, 'conv2_1': 1, 'conv3_1': 1, 'conv4_1': 1, 'conv5_1': 1}

    output = content.clone().requires_grad_(True).to(device)  # Create third image that will be output
    optimizer = optim.Adam([output], lr=learning_rate)  # Iteration Parameters

    for steps in range(1, total_steps + 1):
        optimizer.zero_grad()
        target_features = get_features(output, vgg)  # Get features from target image
        content_loss = torch.mean((target_features['conv5_1'] - content_features['conv5_1']) ** 2)  # Content Loss

        style_loss = 0  # Style Loss - Initialise to 0, then add to it for each layer's gram matrix loss
        for layer in style_weight:
            # Get the target style representation for the layer
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            b, d, h, w = target_feature.shape
            style_gram = style_grams[layer]  # Get the 'style' style representation

            # The style loss for one layer, weighted appropriately
            layer_style_loss = style_weight[layer] * torch.mean((target_gram - style_gram) ** 2)
            style_loss += (layer_style_loss / (d * h * w))  # Add to the style loss

        total_loss = alpha * content_loss + beta * style_loss  # Calculate the total loss

        optimizer.zero_grad()
        total_loss.backward()  # Update target image (back-propagate)
        optimizer.step()

        # Prints iteration number and Total loss at that iteration
        if steps % show_every == 0:
            print("Iteration Number:", steps)
            #print("Style Loss: %.2f" % (style_loss.item()))
            #print("Content Loss: %.2f" % (content_loss.item()))
            print("Total Loss: %.2f" % (total_loss.item()))
        end = time.time()
    # Prints Final Results
    print("------------------------------------------")
    print("FINAL RESULTS")
    print("------------------------------------------")
    print(
        "PARAMETERS:\nTotal Iterations = %d\nNumber of iterations per update = %d\nLearning Rate = %.3f\n"
        "Alpha (Content Weight) = %.5f\nBeta (Style Weight) = %d\n" % (
            total_steps, show_every, learning_rate, alpha, beta))
    print("Runtime of the program = %.2f seconds." % (end - start))
    return image_convert(output)  # Return the output
