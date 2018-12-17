# ElectronBeamWelding
Data Modellings for Experimental Data in Python3

- Contains the following: Artificial Neural Network, Support Vector Regressor, SGD optimised Linear Regressor 
- Ensure you have the following installed on your computer:
1) Python3
2) The following dependencies: PyTorch, Pandas, Numpy, sklearn, Matplotlib
   
  ## To run the ANN
- Go to the file directory containing 'EBW_ANN_Model.py' and 'EBW_ANN_predictor.py' and open command prompt
- Type in ```python``` and press enter
- Type in ```python EBW_ANN_predictor.py``` and press enter

 ## To run the Support Vector Machine
 - Go to the file directory containing "EBW_SVR.py" and open command prompt
 - Type in ```python``` and press enter
 - Type in ```python EBW_SVR.py``` and press enter
  
  ### ANN Architecture
  - The ANN consists of 154 total weights and 3 linear fully connected layers. 
  - The hyperparameters are as follows:
    - Optimizer used is Adam with learning rate at 0.001, betas at 0.9 and 0.999 respectively and epsilon at 0.01
    - Input layer consists of 4 input features and 10 output features
    - Hidden layer consists of 8 output features
    - Output layer consists of 2 output features
    - Rectifier Linear Units were used as activation functions
  - All hyperparameters can be adjusted from 'EBW_ANN_Model.py' in any editor
  
  ### SVM Architecture
  - The SVM is used to perform regression and makes use of Radial Basis Function as a kernel
  - The regularization constant (c) is set at 1e5
  - Gamma is set at 1e5
  - Cache Size was set at 500(mb)
  - SVM settings can be adjusted from 'EBW_SVR.py' in any editor
