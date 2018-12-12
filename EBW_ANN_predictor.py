import torch
import numpy as np
import torch.nn.functional as f
from ElectronBeamWelding.EBW_ANN_Model import Model

net = Model()
net.load_state_dict(torch.load('last_model_state.pth'))
net.eval()
input_arr = [0.0, 0.0, 0.0, 0.0]
input_arr = np.array(input_arr)
input_arr[0] = input("Enter V \n")
input_arr[1] = input("Enter I \n")
input_arr[2] = input("Enter S \n")
input_arr[3] = input("Enter F \n")
input_tensor = torch.tensor(input_arr).float()
inputs = f.normalize(input_tensor, p=2, dim=0)
outputs = net(inputs)
print(outputs)
