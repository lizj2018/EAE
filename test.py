import numpy as np
import matplotlib.pyplot as plt

# import theano # comment / uncomment
import torch
from torch import nn
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
from utils import compute_jacobian, log_det_jacobian

model = nn.Sequential(
    nn.Linear(3, 10),
    nn.ReLU(),
    nn.Linear(10, 3),
    nn.Sigmoid()
    )

x = Variable(torch.rand(100, 3), requires_grad=True)
y = model(x)
y1, y2 = torch.split(y, [1, 2], dim=1)
z = torch.cat([y1, y2], dim=1)
w = model(z)
#z = model(y)
#
# grad_var = torch.zeros(*y.size())
# grad_var[:, 0] = 1
# y.backward(grad_var, retain_graph=True)
# x_grad1 = x.grad.data.numpy().copy()
# zero_gradients(x)
# grad_var.zero_()
# grad_var[:, 0] = 1
# y.backward(grad_var, retain_graph=True)
# x_grad2 = x.grad.data.numpy().copy()
# assert np.allclose(x_grad1, x_grad2)

#j = compute_jacobian(x, y)
#print(j.shape)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

loss_np = []
for i in range(100):
    log_det_j = log_det_jacobian(x, z)
#    log_det_j.requires_grad_ = True
    loss = -torch.mean(log_det_j)
    
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    loss_np.append(loss.data.numpy())
    
plt.figure()
plt.plot(loss_np)
plt.show()


# log_det_j = log_det_jacobian(x, y)
# print(log_det_j.shape)