input_size = X_train.shape[1]  # 4
hidden_size = 10
output_size = 3  # 3 classes in Iris

# Weights & Biases
w1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
w2 = np.random.randn(hidden_size, hidden_size)
b2 = np.zeros((1, hidden_size))
w3 = np.random.randn(hidden_size, output_size)
b3 = np.zeros((1, output_size))


z1 = layer(X_train,w1,b1, activation = 'None')
out1 = layer(X_train,w1,b1,activation ="relu")

z2 = layer(out1, w2,b2,activation = 'None')
out2 = layer(out1, w2, b2, activation = "relu")

z3 = layer(out2, w3,b3,activation = 'None')
out3 = layer(out2, w3,b3, activation = 'softmax')
