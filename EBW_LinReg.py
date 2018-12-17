import numpy as np
import pandas as pd


def dataset_maker():
    """
    This function constructs the randomized data set in two parts, training and testing
    :return: training_set, testing_set
    """
    df_train = pd.read_csv("./training_set.csv")
    df_test = pd.read_csv("./testing_set.csv")
    training_set = df_train.values
    testing_set = df_test.values

    return training_set, testing_set


def step_gradient(m_current, dataset, learning_rate):
    """
    This function performs stochastic gradient descent
    :param m_current: The m matrix
    :param dataset: The dataset
    :param learning_rate: The learning rate
    :return: Updated m and c
    """
    m_gradient = np.zeros((2, 4))
    m_new = np.zeros((2, 4))
    for i in range(len(dataset)):
        x = np.matrix(dataset[i, :4])
        y = np.matrix(dataset[i, 4:6])
        m_gradient += -(2/float(len(dataset)))*np.dot((np.transpose(y) - (np.dot(m_current, np.transpose(x)))), x)
    m_new = m_current - (learning_rate * m_gradient)  # performing update
    return m_new


def cost_function(m_param, dataset, i):
    """
    This function calculates mean square error
    :param m_param: The m matrix
    :param dataset: The dataset
    :param i: The iteration number
    :return: The cost value for iteration i
    """
    return np.around(1/(2*float(len(dataset)))*np.dot(np.transpose(dataset[i, 4:6] - (np.dot(m_param, dataset[i, :4]))),
                                          (dataset[i, 4:6] - (np.dot(m_param, dataset[i, :4])))), decimals=10)


m = np.zeros((2, 4))  # global initial values of m (8x8) and c(8x1)


def training(iterator=800):
    """
    Performs training on the training dataset
    :param iterator: The number of training cycles
    :return: N/A
    """
    dataset, _ = dataset_maker()
    dataset = np.array([[float(data / 100) for data in something] for something in dataset])
    global m
    m = np.zeros((2, 4))
    for i in range(iterator):
        m = step_gradient(m, dataset, learning_rate=0.15)
        # if i % 100 == 0:
        #     print("Iteration: ", i+1, ",Cost = ", cost_function(m, c, dataset, i))
    print("\n", "m = ", m, "\n", "f(x[0]) = ", (np.dot(m, dataset[0, :4]))*100, "y[0]=",
          dataset[0, 4:6]*100)


training()


def testing():
    """
    Performs testing on the testing dataset
    :return: N/A
    """
    print("************************************************************************")
    print("*********************************TESTING********************************")
    print("************************************************************************")
    _, dataset = dataset_maker()
    dataset = np.array([[float(data / 100) for data in something] for something in dataset])
    for i in range(10):
        x = dataset[i, :4]
        y = dataset[i, 4:6]
        print("x = ", x, "\n", "f(x) = ", y * 100, "\n", "predicted f(x) = ", (np.dot(m, x)) * 100, "Cost = ",
              cost_function(m, dataset, i), "\n")


testing()
