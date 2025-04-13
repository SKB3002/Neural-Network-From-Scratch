def gradient_check(X, y, w1, b1, w2, b2, w3, b3, dw1, index):
    epsilon = 1e-5
    i, j = index
    
    original_val = w1[i, j]

    w1[i, j] = original_val + epsilon
    out1_plus = layer(X, w1, b1, 'relu')
    out2_plus = layer(out1_plus, w2, b2, 'relu')
    out3_plus = layer(out2_plus, w3, b3, 'softmax')
    loss_plus = cross_entropy_loss(out3_plus, y)

    w1[i, j] = original_val - epsilon
    out1_minus = layer(X, w1, b1, 'relu')
    out2_minus = layer(out1_minus, w2, b2, 'relu')
    out3_minus = layer(out2_minus, w3, b3, 'softmax')
    loss_minus = cross_entropy_loss(out3_minus, y)

    # Reset w1
    w1[i, j] = original_val

    # Numerical Gradient
    num_grad = (loss_plus - loss_minus) / (2 * epsilon)
    ana_grad = dw1[i, j]

    print(f"--- Gradient Check on w1[{i},{j}] ---")
    print(f"Analytical: {ana_grad:.8f} | Numerical: {num_grad:.8f} | Diff: {abs(ana_grad - num_grad):.8f}")
