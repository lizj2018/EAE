# -*- coding: utf-8 -*-

from EAE import EAE, train

import matplotlib.pyplot as plt
import torch
import numpy as np

N = 300
epochs = 500
lamb = 10.0
x1 = torch.tensor(np.random.randn(N, 1), dtype=torch.float32, requires_grad=True)
x2 = torch.tanh(x1) + torch.randn(N, 1)

plt.figure()
plt.scatter(x1.data.numpy(), x2.data.numpy())
plt.savefig('scatter.png')

params = {'epochs': 500,
          'lr': 1e-3,
          'lamb': 10.0}

x = torch.cat([x1, x2], dim=1)
loss_np0, z_np, x_hat_np = train(x, **params)
    
plt.figure()
plt.scatter(z_np[:,0], z_np[:,1])
plt.show()

plt.figure()
plt.scatter(x_hat_np[:,0], x_hat_np[:,1])
plt.show()
    
print('done x->y')
    
x = torch.cat([x2, x1], dim=1)
loss_np1, z_np, x_hat_np = train(x, **params)
    
plt.figure()
plt.scatter(z_np[:,0], z_np[:,1])
plt.show()

plt.figure()
plt.scatter(x_hat_np[:,0], x_hat_np[:,1])
plt.show()
    
print('done y->x')
    
plt.figure()
plt.plot(loss_np0)
plt.plot(loss_np1)
plt.title('loss0-loss1='+str(loss_np0[-1]-loss_np1[-1]))
plt.savefig('demo.png')
#plt.show()