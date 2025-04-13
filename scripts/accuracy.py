def accuracy(X, y_true):
    z1 = np.dot(X, w1) + b1
    a1 = np.maximum(0, z1)

    z2 =  np.dot(a1, w2) + b2
    a2 = np.maximum(0, z2)

    z3 = np.dot(a2, w3) + b3
    probs = softmax(z3)

    predictions = np.argmax(probs, axis=1)
    labels = np.argmax(y_true, axis=1)

    return np.mean(predictions == labels)

acc = accuracy(X_test, y_test)
print(f"\nTest Accuracy: {acc * 100:.2f}%")
