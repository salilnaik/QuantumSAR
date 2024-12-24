import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.metrics import confusion_matrix, accuracy_score
from quanv_dataset import QuanvDataset
from torch.utils.data.sampler import SubsetRandomSampler



batch_size = 128
num_epochs = 200
print_every = 10
init_lr = 0.0001

print(f"batch size {batch_size}\tnum epochs {num_epochs}\tinit_lr {init_lr}")

pil = v2.Compose([
    v2.Grayscale(num_output_channels=1),
    v2.PILToTensor(),
    v2.ToDtype(torch.float32),
    v2.CenterCrop(size=128),
    v2.Normalize(mean=[30.761015], std=[25.408434])
])

device = torch.device("cuda")

class Net(nn.Sequential):
    def __init__(self):
        super(Net, self).__init__(
        nn.Conv2d(1,4, kernel_size=2),
        nn.Conv2d(4,32, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout2d(0.25),
        nn.Flatten(),
        nn.Linear(30752, 128),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(128, 10)
        )

    def forward(self, x):
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.drop1(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = self.fc2(x)
        return x

def main(path, test_path):
    train_dataset = QuanvDataset(path)
    test_dataset = QuanvDataset(test_path)
    print(train_dataset)
    dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    model = Net()
    model = model.to(device)
        
    loss_function = torch.nn.CrossEntropyLoss()
    train_acc = 0
    torch.autograd.set_detect_anomaly(True)         
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    for epoch in range(num_epochs):
        for id_batch, (X_batch, y_batch) in enumerate(dataloader):
            pred_y = model(X_batch.to(device))
            loss = loss_function(pred_y.to(device), F.one_hot(y_batch.to(device), num_classes=10).float())
            model.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        if(epoch%print_every == 0):
            print(f"Epoch {epoch}/{num_epochs}\tLoss {round(loss.data.item(), 6)}")
    model.eval()

    cm = []
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    for x,y in test_dataloader:
        outputs = model(x.to(device))
        prediction = torch.round(torch.argmax(outputs.to(device), dim=1))
        cm.append(confusion_matrix(y.cpu().numpy(), prediction.cpu().numpy(), labels=[i for i in range(0,10)]))

    finalcm = np.sum(cm, axis=0)
    acc = np.sum([finalcm[i][i] for i in range(0,10)])/np.sum(cm)

    print(finalcm)
    print([list(i) for i in finalcm])
    print(acc)

main("./quanv_images1/train/", "./quanv_images1/test/")
main("./quanv_images2/train/", "./quanv_images2/test/")
main("./quanv_images3/train/", "./quanv_images3/test/")
main("./quanv_images4/train/", "./quanv_images4/test/")
main("./quanv_images5/train/", "./quanv_images5/test/")
