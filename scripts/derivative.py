
def relu_derivative(z):
    return np.where(z > 0, 1, 0)  # Return 1 if z > 0, else 0


def softmax(z):
    exp_x = np.exp(z - np.max(z, axis=1, keepdims=True))  # for numerical stability
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def softmax_derivative(z, y_true):
    softmax_output = softmax(z)

    return softmax_output - y_true
