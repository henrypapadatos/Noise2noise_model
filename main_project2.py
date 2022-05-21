
import torch
from torch import nn
from Miniproject_2 import model

input_tensor = torch.normal(0, 1, size=(3,2,2), requires_grad=True)
target = torch.normal(0, 1, size=(3,2,2), requires_grad=True)

sequential_torch = nn.Sequential(nn.ReLU(),nn.Sigmoid())

criterion = nn.MSELoss()

y = sequential_torch(input_tensor)

loss = criterion(y, target)

# loss_val = loss(y, target)

loss.backward()
with torch.no_grad():
    loss.grad


# loss = MSE()

# our_loss = loss.forward(input_tensor, target)
# torch_loss = nn.MSELoss()

# torch_loss_val = torch_loss(input_tensor, target)

# print(our_loss-torch_loss_val)

# relu = Relu()
# sigmoid = Sigmoid()
# sequential = Seq(relu,sigmoid)

# sequential_torch = nn.Sequential(nn.ReLU(),nn.Sigmoid())


# y = sequential.forward(input_tensor)

# y_torch = sequential_torch(input_tensor)

# print(y-y_torch)

# y = sequential.backward(y)

# # y_torch.backward(gradient)
# print(y-torch.gradient(y_torch))
