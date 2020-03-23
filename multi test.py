#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
from torchvision import datasets, transforms


# In[2]:


# Hyper parameters
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

num_clients = 10

num_processes = 2
device = "cuda"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[3]:


class Model(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(Model, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


# In[4]:


train_data = datasets.MNIST(
    "../data",
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
)
test_data = datasets.MNIST(
    "../data",
    train=False,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
)


# In[5]:


# Data loader
train_loader = torch.utils.data.DataLoader(
    dataset=train_data, batch_size=batch_size, num_workers=2, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_data, batch_size=batch_size, num_workers=2, shuffle=False
)


# In[6]:


# Loss and optimizer
model = Model()  # .to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# In[9]:


import torch.multiprocessing as mp


def train(model_old):
    # Construct data_loader, optimizer, etc.
    model = Model()
    model.load_state_dict(model_old.state_dict())
    model.to(device)
    for i, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        loss_fn(model(data.to(device)), labels.to(device)).backward()
        optimizer.step()  # This will update the shared parameters
        # if i > 100:
        #     break
    return model_old.state_dict()


# In[10]:



# NOTE: this is required for the ``fork`` method to work
model.share_memory()
processes = []
for rank in range(num_processes):
    p = mp.Process(target=train, args=(model,))
    p.start()
    processes.append(p)
for p in processes:
    p.join()


# In[11]:

