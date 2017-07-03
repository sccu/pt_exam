import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [1, 0, 0, 1]

class Xor(nn.Module):
  def __init__(self):
    super(Xor, self).__init__()
    self.fc1 = nn.Linear(2, 50)
    self.fc2 = nn.Linear(50, 1)

  def forward(self, x):
    l1 = self.fc1(x)
    nl = F.relu(l1)
    l2 = self.fc2(nl)
    return l2

xor = Xor()
input = Variable(torch.FloatTensor(X))
target = Variable(torch.FloatTensor(y))

# xor.zero_grad()
optimizer = optim.SGD(xor.parameters(), lr=0.001)
optimizer.zero_grad()
criterion = nn.MSELoss()
for i in range(5000):
  out = xor(input)
  loss = criterion(out, target)
  loss.backward()
  optimizer.step()
  if i % 100 == 0:
    print(out)

