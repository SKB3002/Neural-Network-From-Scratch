def cross_entropy_loss(predictions, y_true):
    # Add epsilon for numerical stability
    eps = 1e-15
    predictions = np.clip(predictions, eps, 1 - eps)
    loss = -np.mean(np.sum(y_true * np.log(predictions + 1e-9), axis=1))
    return loss
