import pandas as pd
import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.in_layer = nn.Linear(in_features=4, out_features=10)
        self.hidden_layer_1 = nn.Linear(in_features=10, out_features=8)
        self.out_layer = nn.Linear(in_features=8, out_features=2, bias=False)

    def forward(self, x):
        x = F.relu(self.in_layer(x))
        x = F.relu(self.hidden_layer_1(x))
        x = self.out_layer(x)
        return x


net = Model()
net_total_params = sum(p.numel() for p in net.parameters())
print(net_total_params)


# training
def train():
    start_time = time.time()
    EPOCH = 800
    TRAIN_PATH = "./Prediction_EBW_csv.csv"
    DataFrame = pd.read_csv(TRAIN_PATH)
    DataFrame = DataFrame.sample(frac=1).reset_index(drop=True)
    # converting data to tensors
    data_array = DataFrame.values
    data_tensor = torch.from_numpy(data_array)
    input_indices = torch.tensor([1, 2, 3, 4])
    output_indices = torch.tensor([5, 6])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999), eps=0.01)
    for epoch in range(EPOCH):
        running_loss = 0.0
        for i, data in enumerate(data_tensor, 0):
            inputs = torch.index_select(data, 0, input_indices).float()
            inputs = F.normalize(inputs, p=2, dim=0)
            truth = torch.index_select(data, 0, output_indices).float()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, truth)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            predictor = outputs
        print('EPOCH: %d, LOSS: %.5f' % (epoch + 1, running_loss/float(len(data_array))))
    print("Finished Training. Time taken: ", (time.time()-start_time))
    torch.save(net.state_dict(), 'last_model_state.pth')


# train()


# testing
def test():
    start_time = time.time()
    TEST_PATH = "./testing_set.csv"
    df_test = pd.read_csv(TEST_PATH)
    test_data_array = df_test.values
    test_data_tensor = torch.from_numpy(test_data_array)
    test_input_indices = torch.tensor([1, 2, 3, 4])
    test_output_indices = torch.tensor([5, 6])
    net = Model()
    net.load_state_dict(torch.load('last_model_state.pth'))
    net.eval()
    print("GROUND TRUTH\t\tPREDICTED\t\tRMSE\t\tMAPE\nBW\t\tBP\t\t\tBW\t\tBP")
    with torch.no_grad():
        total_rmse = 0.0
        total_mape = 0.0
        total_i = 0.0
        for i in enumerate(test_data_tensor):
            inputs = torch.index_select(i[1], 0, test_input_indices).float()
            inputs = F.normalize(inputs, p=2, dim=0)
            outputs = net(inputs)
            truth = torch.index_select(i[1], 0, test_output_indices).float()
            predicted = outputs.numpy()
            rmse = np.sqrt(((predicted - truth) ** 2).mean())
            total_rmse += rmse
            total_i += 1.0
            truth = truth.numpy()
            mape = np.mean(np.abs((truth - predicted) / truth)) * 100
            total_mape += mape
            print("%.2f\t%.2f" % (truth[0], truth[1]), "\t\t%.2f\t%.2f\t%.3f\t\t%.3f%%" % (predicted[0], predicted[1],
                                                                                           rmse, mape), "\n")
        print("Time taken: %.5f" % (time.time() - start_time))
        print("Average RMSE: %.5f" % (total_rmse/total_i))
        print("Average MAPE: %.5f" % (total_mape/total_i))


test()
