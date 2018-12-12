import numpy as np
import pandas as pd
import torch
import torch.nn.functional as f
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# #############################################################################
# Prepare training data
DataFrame = pd.read_csv("./training_set.csv")
data_array = DataFrame.values
data_tensor = torch.from_numpy(data_array)
input_indices = torch.tensor([1, 2, 3, 4])
input_data_tensor = torch.index_select(data_tensor, 1, input_indices)
input_data_tensor = f.normalize(input_data_tensor, p=2, dim=1)
input_arr = input_data_tensor.numpy()
X = input_arr.reshape(len(input_arr)*2, -1)
output_indices = torch.tensor([5, 6])
output_data_tensor = torch.index_select(data_tensor, 1, output_indices)
BW_index, BP_index = torch.tensor([0]), torch.tensor([1])
BW_arr = torch.index_select(output_data_tensor, 1, BW_index).numpy()
BP_arr = torch.index_select(output_data_tensor, 1, BP_index).numpy()
y = np.asarray(list(output_data_tensor.numpy().flat))

#############################################################################
# Fit regression model
svr_rbf = SVR(kernel='rbf', C=1e5, gamma=1e5, cache_size=500)
svr_rbf.fit(X, y)
y_rbf = svr_rbf.predict(X)
print(svr_rbf.fit(X, y).score(X, y))

# #############################################################################
# Look at the results
lw = 1
data = plt.scatter(X[:, [0]], y, color='darkorange', label='inputs', s=10)
rbf, _ = plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend(handles=[data, rbf])
plt.show()

# #############################################################################
# test on user input
user_input_arr = np.array([0.0, 0.0, 0.0, 0.0])
user_input_arr[0] = input("Enter V \n")
user_input_arr[1] = input("Enter I \n")
user_input_arr[2] = input("Enter S \n")
user_input_arr[3] = input("Enter F \n")
user_input_tensor = torch.tensor(user_input_arr).float()
user_input_tensor = f.normalize(user_input_tensor, p=2, dim=0)
user_input_arr = user_input_tensor.numpy()
user_X = user_input_arr.reshape(int(len(user_input_arr)/2), -1)
prediction = svr_rbf.predict(user_X)
print(prediction)
