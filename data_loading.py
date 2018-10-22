import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def load_mnist(directory='./data'):
    """
    Load MNIST as PyTorch tensors and cache the dataset in the specified directory.

    :return: X_train, y_train, X_test, y_test
    """
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    train_set = datasets.MNIST(root=directory, train=True, download=True, transform=trans)
    test_set = datasets.MNIST(root=directory, train=False, download=True, transform=trans)
    X_train = train_set.train_data.to(dtype=torch.float)
    y_train = train_set.train_labels
    X_test = test_set.test_data.to(dtype=torch.float)
    y_test = test_set.test_labels
    return X_train, y_train, X_test, y_test
