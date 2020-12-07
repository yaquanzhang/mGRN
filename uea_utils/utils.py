
def normalization(X_train, X_test):
    # perform normalization to X_train and X_test
    for i in range(X_train.shape[2]):
        this_mean = X_train[:, :, i].mean()
        this_std = X_train[:, :, i].std()
        X_train[:, :, i] = (X_train[:, :, i] - this_mean) / this_std
        X_test[:, :, i] = (X_test[:, :, i] - this_mean) / this_std
    return X_train, X_test