import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import pennylane as qml
import numpy as np
from pennylane.templates import RandomLayers
import os

batch_size = 1

pil = v2.Compose([
    v2.Grayscale(num_output_channels=1),
    v2.PILToTensor(),
    v2.ToDtype(torch.float32),
    v2.CenterCrop(size=128),
    v2.Normalize(mean=[30.761015], std=[25.408434])
])
train_dataset = ImageFolder(root="./MSTAR-10-Classes/train", transform=pil, target_transform=v2.ToDtype(torch.float32))
dataloader = DataLoader(train_dataset, shuffle=True)


n_layers = 2 # number of random layers
kernel_size = 2
stride = 1

dev = qml.device("default.qubit", wires=4, shots=20)
# Random circuit parameters
rand_params = np.random.uniform(high=2 * np.pi, size=(n_layers, 4))

@qml.qnode(dev)
def circuit(phi):
    # Encoding of 4 classical input values
    for j in range(4):
        qml.RY(np.pi * phi[j], wires=j)

    # Random quantum circuit
    RandomLayers(rand_params, wires=list(range(4)))

    # Measurement producing 4 classical output values
    return qml.counts()

def quanv(image):
    """Convolves the input image with many applications of the same quantum circuit."""
    out = np.zeros((128-kernel_size+1, 128-kernel_size+1))

    # Loop over the coordinates of the top-left pixel of 2X2 squares
    for j in range(0, 127, stride):
        for k in range(0, 127, stride):
            # Process a squared 2x2 region of the image with a quantum circuit
            q_results = circuit(
                [
                    image[j, k],
                    image[j, k + 1],
                    image[j + 1, k],
                    image[j + 1, k + 1]
                ]
            )
            # Assign expectation values to different channels of the output pixel (j/2, k/2)
            out[j,k] = max(q_results).count("1")/4
    return out


for i, x in enumerate([train_dataset[0]]):
    k = train_dataset[i][0].numpy().reshape(128,128) / 255.0

    q = quanv(k)

    path = r"quanv_images/train"
    name = ""
    match train_dataset[i][1]:
        case 0:
            name = "2S1"
        case 1:
            name = "BMP2"
        case 2:
            name = "BRDM2"
        case 3:
            name = "BTR60"
        case 4:
            name = "BTR70"
        case 5:
            name = "D7"
        case 6:
            name = "T62"
        case 7:
            name = "T72"
        case 8:
            name = "ZIL131"
        case 9:
            name = "ZSU_23_4"
    
    pat = os.path.join(path, f"{name}/{name}_{i}.npy")
    np.save(pat, q)
    