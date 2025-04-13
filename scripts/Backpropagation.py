# Backward Pass
    dz3 = softmax_derivative(z3, y_train)  # (batch, 3)
    dw3 = np.dot(out2.T, dz3)
    db3 = np.sum(dz3, axis=0, keepdims=True)

    dz2 = np.dot(dz3, w3.T) * relu_derivative(z2)
    dw2 = np.dot(out1.T, dz2)
    db2 = np.sum(dz2, axis=0, keepdims=True)

    dz1 = np.dot(dz2, w2.T) * relu_derivative(z1)
    dw1 = np.dot(X_train.T, dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)

    # Update weights
    w3 -= lr * dw3
    b3 -= lr * db3
    w2 -= lr * dw2
    b2 -= lr * db2
    w1 -= lr * dw1
    b1 -= lr * db1
