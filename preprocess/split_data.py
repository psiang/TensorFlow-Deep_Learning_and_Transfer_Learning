from sklearn.model_selection import train_test_split


def split(data, label, rate=0.2):
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=rate)
    return x_train, y_train, x_test, y_test
