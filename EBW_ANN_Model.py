import pandas as pd
import torch
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
    total_BW_error = 0.0
    total_BP_error = 0.0
    EPOCH = 50
    # TRAIN_PATH = "./training_set.csv"
    TRAIN_PATH = "./Prediction_EBW_csv.csv"
    DataFrame = pd.read_csv(TRAIN_PATH)
    DataFrame = DataFrame.sample(frac=1).reset_index(drop=True)
    # converting data to tensors
    data_array = DataFrame.values
    data_tensor = torch.from_numpy(data_array)
    input_indices = torch.tensor([1, 2, 3, 4])
    output_indices = torch.tensor([5, 6])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=0.01)
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
            predicted = predictor.detach().numpy()
            print("GroundTruth: BW: %.2f, BP: %.2f" % (truth[0], truth[1]), " Predicted: BW: %.2f, BP: %.2f" %
                  (predicted[0], predicted[1]), "\n")
            percentage_error_BW = (abs((truth[0].numpy().astype(float) - predicted[0].astype(float))) / truth[0].numpy()
                                   .astype(float)) * 100
            percentage_error_BP = (abs((truth[1].numpy().astype(float) - predicted[1].astype(float))) / truth[1].numpy()
                                   .astype(float)) * 100
            print("Percentage error in BW: %.2f%%, Percentage error in BP: %.2f%%" %
                  (percentage_error_BW, percentage_error_BP), "\n")
        print('EPOCH: %d, LOSS: %.5f' % (epoch + 1, running_loss/81.0))
    print("Finished Training")
    torch.save(net.state_dict(), 'last_model_state.pth')


# train()


# testing
def test():
    total_BW_error = 0.0
    total_BP_error = 0.0
    TEST_PATH = "./testing_set.csv"
    df_test = pd.read_csv(TEST_PATH)
    test_data_array = df_test.values
    test_data_tensor = torch.from_numpy(test_data_array)
    test_input_indices = torch.tensor([1, 2, 3, 4])
    test_output_indices = torch.tensor([5, 6])
    net = Model()
    with torch.no_grad():
        for i in enumerate(test_data_tensor):
            percentage_error_BW = 0.0
            percentage_error_BP = 0.0
            inputs = torch.index_select(i[1], 0, test_input_indices).float()
            inputs = F.normalize(inputs, p=2, dim=0)
            outputs = net(inputs)
            truth = torch.index_select(i[1], 0, test_output_indices).float()
            predicted = outputs.numpy()
            print("GroundTruth: BW: %.2f, BP: %.2f" % (truth[0], truth[1]), " Predicted: BW: %.2f, BP: %.2f" %
                  (predicted[0], predicted[1]), "\n")
            percentage_error_BW = (abs((truth[0].numpy().astype(float) - predicted[0].astype(float))) / truth[0].numpy()
                                   .astype(float)) * 100
            percentage_error_BP = (abs((truth[1].numpy().astype(float) - predicted[1].astype(float))) / truth[1].numpy()
                                   .astype(float)) * 100
            print("Percentage error in BW: %.2f%%, Percentage error in BP: %.2f%%" %
                  (percentage_error_BW, percentage_error_BP), "\n")
            total_BW_error += percentage_error_BW
            total_BP_error += percentage_error_BP
        print("Average BW error: ", total_BW_error / 25.0, "%")
        print("Average BP error: ", total_BP_error / 25.0, "%")
        # print("Average BW Accuracy: ", 100 - (total_BW_error / 25.0), "%")
        # print("Average BP Accuracy: ", 100 - (total_BP_error / 25.0), "%")


# test()
