import torch
import torch.nn.functional as F

# class SimpleFCN(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = torch.nn.Sequential(
#             torch.nn.Flatten(),
#             torch.nn.Linear(28 * 28, 512),
#             torch.nn.ReLU(),
#             torch.nn.Linear(512, 256),
#             torch.nn.ReLU(),
#             torch.nn.Linear(256, 10),
#         )

#     def forward(self, x):
#         x = self.model(x)
#         return x


# # N-layer fully connected network

# class SimpleFCN(torch.nn.Module):      # simple-2layer
#     def __init__(self, num_classes=10):
#         super(SimpleFCN, self).__init__()
#         self.fc1 = torch.nn.Linear(3 * 128 * 128, 256)
#         self.fc2 = torch.nn.Linear(256, num_classes)

#     def forward(self, x):
#         x = x.view(x.size(0), -1)  # Flatten the input
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
    
# class SimpleFCN(torch.nn.Module):         # simple-3layer
#     def __init__(self, num_classes=10):
#         super(SimpleFCN, self).__init__()
#         self.fc1 = torch.nn.Linear(3 * 128 * 128, 512)
#         self.fc2 = torch.nn.Linear(512, 256)
#         self.fc3 = torch.nn.Linear(256, num_classes)

#     def forward(self, x):
#         x = x.view(x.size(0), -1)  # Flatten the input
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
    
# class SimpleFCN(torch.nn.Module):            # simple-4layer
#     def __init__(self, num_classes=10):
#         super(SimpleFCN, self).__init__()
#         self.fc1 = torch.nn.Linear(3 * 128 * 128, 512)
#         self.fc2 = torch.nn.Linear(512, 512)
#         self.fc3 = torch.nn.Linear(512, 256)
#         self.fc4 = torch.nn.Linear(256, num_classes)

#     def forward(self, x):
#         x = x.view(x.size(0), -1)  # Flatten the input
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = self.fc4(x)
#         return x
    
# class SimpleFCN(torch.nn.Module):        # simple-6layer
#     def __init__(self, num_classes=10):
#         super(SimpleFCN, self).__init__()
#         self.fc1 = torch.nn.Linear(3 * 128 * 128, 512)
#         self.fc2 = torch.nn.Linear(512, 512)
#         self.fc3 = torch.nn.Linear(512, 512)
#         self.fc4 = torch.nn.Linear(512, 512)
#         self.fc5 = torch.nn.Linear(512, 256)
#         self.fc6 = torch.nn.Linear(256, num_classes)

#     def forward(self, x):
#         x = x.view(x.size(0), -1)  # Flatten the input
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = F.relu(self.fc4(x))
#         x = F.relu(self.fc5(x))
#         x = self.fc6(x)
#         return x
    
# class SimpleFCN(torch.nn.Module):        # simple-8layer
#     def __init__(self, num_classes=10):
#         super(SimpleFCN, self).__init__()
#         self.fc1 = torch.nn.Linear(3 * 128 * 128, 512)
#         self.fc2 = torch.nn.Linear(512, 512)
#         self.fc3 = torch.nn.Linear(512, 512)
#         self.fc4 = torch.nn.Linear(512, 512)
#         self.fc5 = torch.nn.Linear(512, 512)
#         self.fc6 = torch.nn.Linear(512, 512)
#         self.fc7 = torch.nn.Linear(512, 256)
#         self.fc8 = torch.nn.Linear(256, num_classes)

#     def forward(self, x):
#         x = x.view(x.size(0), -1)  # Flatten the input
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = F.relu(self.fc4(x))
#         x = F.relu(self.fc5(x))
#         x = F.relu(self.fc6(x))
#         x = F.relu(self.fc7(x))
#         x = self.fc8(x)
#         return x


# a extremely small network only has 3.5k parameters
class SimpleFCN(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleFCN, self).__init__()
        self.conv = torch.nn.Conv2d(3, 3, kernel_size=5, stride=4, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=4, stride=4)
        # Now spatial dimensions are reduced to 8x8
        self.fc1 = torch.nn.Linear(3 * 8 * 8, 16)
        self.fc2 = torch.nn.Linear(16, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x