from engine import Value
from nn import *

x = [[0.0, 0.0], [1.5, 1.5], [0.2, 0.2], [0.5, 0.1], [1.5, 0.5], [5.0, 1.0], [0.6, 0.2], [1.0, 0.4], [1.7, 0.5], [2.0, 0.7]]
y = [1, 0, 1, 1, 0, 0, 1, 0, 0, 0]
nn = DenseNN(2, (3, 3, 1), ('tanh', 'tanh', 'sigmoid'))
nn = nn.Train(x, y, loss='BCE', epochs_number=20, learning_rate=0.3, batch_size=5)
print(nn.Predict([1.0, 1.0]))