import numpy as np
import pandas as pd
import torch
import time
import torch.nn.functional as f
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# #############################################################################
# Prepare training data
DataFrame = pd.read_csv("./Prediction_EBW_csv.csv")
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
# test on user input
start_time = time.time()
TEST_PATH = "./testing_set.csv"
df_test = pd.read_csv(TEST_PATH)
test_data_array = df_test.values
test_data_tensor = torch.from_numpy(test_data_array)
test_input_indices = torch.tensor([1, 2, 3, 4])
test_output_indices = torch.tensor([5, 6])
print("GROUND TRUTH\t\tPREDICTED\t\tRMSE\t\tMAPE\nBW\t\tBP\t\t\tBW\t\tBP")
total_rmse = 0.0
total_mape = 0.0
total_i = 0.0
for i in enumerate(test_data_tensor):
    inputs = torch.index_select(i[1], 0, test_input_indices).float()
    inputs = f.normalize(inputs, p=2, dim=0)
    test_X = inputs.reshape(int(len(inputs)/2), -1)
    predicted = svr_rbf.predict(test_X)
    truth = torch.index_select(i[1], 0, test_output_indices).float()
    rmse = np.sqrt(((predicted - truth) ** 2).mean())
    total_rmse += rmse
    total_i += 1.0
    truth = truth.numpy()
    mape = np.mean(np.abs((truth - predicted) / truth)) * 100
    total_mape += mape
    print("%.2f\t%.2f" % (truth[0], truth[1]), "\t\t%.2f\t%.2f\t%.3f\t\t%.3f%%" % (predicted[0], predicted[1], rmse, mape), "\n")
print("Time taken: %.5f" % (time.time() - start_time))
print("Average RMSE: %.5f" % (total_rmse/total_i))
print("Average MAPE: %.5f" % (total_mape/total_i))

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

