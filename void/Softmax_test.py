import torch
import copy

w = torch.zeros(4)
w[0] = 1
ws = torch.softmax(w, 0)
wws = ws*w
print(w, ws, torch.sum(wws))



net = torch.nn.Sequential(torch.nn.Linear(8,4), torch.nn.Linear(4,2))
backup = copy.deepcopy(net.state_dict())
print(backup)