def layer(inputs, weights, bias, activation):
    z = np.dot(inputs, weights) + bias
    if activation == "sigmoid":
        return 1 / (1 + np.exp(-z))
    elif activation == "relu":
        return np.maximum(0, z)
    elif activation == 'softmax':
        exp_x = np.exp(z - np.max(z, axis=1, keepdims=True))  # for numerical stability
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    else:
        return z  # Linear (no activation)
