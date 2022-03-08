import pathlib
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import torch
import numpy as np
import skimage.transform as skit

# image = Image.open("images/zebra.jpg")
image = Image.open(r"C:\Users\krist\Documents\NTNU\V2022\TDT4265\assigmnents\TDT4265_StarterCode\assignment3\images\zebra.jpg")
print("Image shape:", image.size)

model = torchvision.models.resnet18(pretrained=True)
print(model)
first_conv_layer = model.conv1
print("First conv layer weight shape:", first_conv_layer.weight.shape)
print("First conv layer:", first_conv_layer)

# Resize, and normalize the image with the mean and standard deviation
image_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image = image_transform(image)[None]
print("Image shape:", image.shape)

activation = first_conv_layer(image)
print("Activation shape:", activation.shape)


def torch_image_to_numpy(image: torch.Tensor):
    """
    Function to transform a pytorch tensor to numpy image
    Args:
        image: shape=[3, height, width]
    Returns:
        iamge: shape=[height, width, 3] in the range [0, 1]
    """
    # Normalize to [0 - 1.0]
    image = image.detach().cpu() # Transform image to CPU memory (if on GPU VRAM)
    image = image - image.min()
    image = image / image.max()
    image = image.numpy()
    if len(image.shape) == 2: # Grayscale image, can just return
        return image
    assert image.shape[0] == 3, "Expected color channel to be on first axis. Got: {}".format(image.shape)
    image = np.moveaxis(image, 0, 2)
    return image

# // Task4b // =====================================================
def run_task4b():
    indices = [14, 26, 32, 49, 52]

    fig = plt.figure(figsize=(20,10))

    for i, indice in enumerate(indices):
        kernel = model.conv1.weight[indice, :, :, :]
        activation = model.conv1.forward(image)[0, indice, :, :]
        
        fig.add_subplot(2, len(indices), i+1)
        fig.add_subplot(2, len(indices), i+1).set_title(str(indice))
        plt.imshow(torch_image_to_numpy(kernel))
        
        fig.add_subplot(2, len(indices), len(indices)+i+1)
        plt.imshow(torch_image_to_numpy(activation), cmap="gray")

    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)

    plt.savefig(plot_path.joinpath(f"task4b_plot.png"))
    plt.show()
# run_task4b()

# // Task4c // ================================================
def run_task4c():
    model = torchvision.models.resnet18(pretrained=True)
    model_4c = torch.nn.Sequential(*list(model.children())[:-2])
    activations = model_4c.forward(image)

    fig = plt.figure(figsize=(100, 100))

    for i in range(10):  
        if i < 5:  
            fig.add_subplot(2, 10, i+1)
            fig.add_subplot(2, 10, i+1).set_title(str(i))
            plt.imshow(torch_image_to_numpy(activations[0, i, :, :]), cmap="gray") 
        else:
            fig.add_subplot(1, 10, i-5+1)
            fig.add_subplot(1, 10, i-5+1).set_title(str(i))
            plt.imshow(torch_image_to_numpy(activations[0, i, :, :]), cmap="gray") 

    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    fig.tight_layout(pad=20)

    plt.savefig(plot_path.joinpath(f"task4c_plot.png"))
    plt.show()
# run_task4c()
