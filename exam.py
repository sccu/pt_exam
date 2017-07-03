from __future__ import print_function
import torch
from torch.autograd import Variable

x = Variable(torch.ones(2, 2), requires_grad=True)
print(x.requires_grad)
print(x)

y = x + 2
print(y.creator)

z = y * y * 3
out = z.mean()
print(out)

print(x.grad)
out.backward()
print(x.grad)

