def cross_entropy_loss_with_regularization(y_pred, y_true, model_weights, lambda_reg):
    m = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / m
    
    # L2 Regularization Term
    l2_reg = lambda_reg * np.sum(np.square(model_weights))  # Sum of squares of all weights
    
    total_loss = loss + l2_reg  # Total loss with regularization
    return total_loss
