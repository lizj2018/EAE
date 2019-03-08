import torch
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import grad
import numpy as np

EPS = 1e-8

def jacobian(inputs, outputs):
    return torch.stack([grad(outputs[:, i].sum(), inputs, retain_graph=True, 
                              create_graph=True)[0]
                        for i in range(outputs.size(1))], dim=1)

def compute_jacobian(inputs, output):
    assert inputs.requires_grad
    num_classes = output.size()[1]

    jacobian = torch.zeros(num_classes, *inputs.size(), dtype=torch.float)
    grad_output = torch.zeros(*output.size(), dtype=torch.float)
    if inputs.is_cuda:
        grad_output = grad_output.cuda()
        jacobian = jacobian.cuda()

    for i in range(num_classes):
        zero_gradients(inputs)
        grad_output.zero_()
        grad_output[:, i] = 1
        output.backward(grad_output, retain_graph=True)
        # print(type(inputs.grad))
        jacobian[i] = inputs.grad

    return torch.transpose(jacobian, dim0=0, dim1=1)

def log_det_jacobian(inputs, outputs):
    batch_size = inputs.size()[0]
#    j = compute_jacobian(inputs, outputs)
    j = jacobian(inputs, outputs)
    det_j = torch.zeros(batch_size, 1)

    for i in range(batch_size):
        det_j[i] = torch.abs(torch.det(j[i]))
    return torch.log(EPS + det_j)

def test():
    x = torch.tensor(np.random.randn(5, 3), dtype=torch.float, requires_grad=True)
    y = torch.log(x)[:, :2]
    # y = x
    j = jacobian(x, y)
#    log_det_j = log_det_jacobian(x, y)
    print(j)
    print(j.shape)
#    print(log_det_j)
#    print(log_det_j.shape)

def check():
    x = torch.tensor(np.random.randn(5, 2), dtype=torch.float, requires_grad=True)
    y = torch.log(x)
    j_t = torch.zeros(5, 2, 2)
    for i in range(5):
        j_t[i, :, :] = torch.diagflat(1/x[i, :])

    print(j_t)
    j = compute_jacobian(x, y)
    print(j)
    # assert(j_t.data.numpy() == j.data.numpy())

    log_det_j_t = torch.log(torch.abs(1/x[:, 0]*1/x[:, 1])).reshape(-1, 1)
    log_det_j = log_det_jacobian(x, y)

    print(log_det_j_t)
    print(log_det_j)
    # assert (log_det_j.data.numpy() == log_det_j_t.data.numpy())
    # print(log_det_j.shape)

#check()
#test()