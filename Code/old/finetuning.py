import torch
import torch.utils.data
import torchvision
from torchvision import transforms
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn

import torch_utils

device = torch_utils.get_gpu()

######## START PARAMS #########
batch_size = 16
max_epochs = 1
######### END PARAMS ##########

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

print(len(train_loader))

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

train_iterator = torch_utils.create_iterator(train_loader)

# x,t = next(train_iterator)
# x,t = x.to(device), t.to(device)

# print("Label Indexes", t)
# print([classes[ti] for ti in t])

# plt.grid(False)
# plt.imshow(torchvision.utils.make_grid(x).cpu().data.permute(0,2,1).contiguous().permute(2,1,0), cmap=plt.cm.binary) # type: ignore
# plt.show()

model = resnet18(weights=None)
opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

model.to(device)
model.train()

for epoch in range(max_epochs):
    running_loss = 0

    for i, data in enumerate(train_loader, 0):
        inp, labels = data
        inp = inp.to(device)
        labels = labels.to(device)
        opt.zero_grad()
        predictions = model(inp)
        loss = criterion(predictions, labels)
        loss.backward()
        opt.step()

        running_loss += loss.item()

        if i % 2000 == 1999:
            print(f"{epoch + 1}, {i + 1:5d} loss: {running_loss / 2000:.3f}")
            running_loss = 0

torch.save(model.state_dict(), "./models/cifar_ot1.pth")

model.eval()

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        # calculate outputs by running images through the network
        outputs = model(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')