import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as sm
import pickle
from mlxtend.data import loadlocal_mnist

np.random.seed(4)


def safe_log(x):
    x[x == 0] = 1
    return np.log(x)


# returns product of all elements
def product(a):
    prod = 1
    for x in a:
        prod *= x
    return prod


def get_labels(y):
    return np.argmax(y, axis=1).ravel()


def cross_entropy_loss(y_true, y_pred):
    return -np.average(np.sum(y_true * safe_log(y_pred), axis=1))


def accuracy(y_true, y_pred):
    return sm.accuracy_score(get_labels(y_true), get_labels(y_pred))


def f1_score(y_true, y_pred):
    return sm.f1_score(get_labels(y_true), get_labels(y_pred), average='macro')


# list to one hot encoded numpy array
def one_hot_encode(a):
    b = np.zeros((len(a), max(a) + 1))
    b[range(len(a)), a] = 1
    return b


class DataLoader:
    def __init__(self, batch_size, x_train, y_train, x_test, y_test):
        self.batch_size = batch_size
        self.cur = 0
        self.x_train = x_train
        self.y_train = y_train
        m = len(x_test) // 2
        # m = len(x_test)
        self.x_val = x_test[:m]
        self.y_val = y_test[:m]
        self.x_test = x_test[m:]
        self.y_test = y_test[m:]

    def shape(self):
        return self.x_train[0].shape

    def reset(self):
        self.cur = 0

    def next(self):
        return self.cur < len(self.x_train)

    def next_train_batch(self):
        end = self.cur + self.batch_size
        x_train_batch = self.x_train[self.cur: end]
        y_train_batch = self.y_train[self.cur: end]
        self.cur = end
        return x_train_batch, y_train_batch

    def train_data(self):
        return self.x_train, self.y_train

    def val_data(self):
        return self.x_val, self.y_val

    def test_data(self):
        return self.x_test, self.y_test


class ToyDataLoader(DataLoader):
    @staticmethod
    def read_data(data_file):
        df = pd.read_csv(data_file, delim_whitespace=True, header=None)
        df = df.sample(frac=1)  # shuffle data
        x = np.array(df.iloc[:, :-1])
        x = x.astype(float) / np.max(x)
        y = np.array(df.iloc[:, -1])
        y = np.array(pd.get_dummies(y))
        print(x.shape, y.shape)
        return x, y

    def __init__(self, batch_size):
        x_train, y_train = self.read_data('sir-toy-dataset/trainNN.txt')
        x_test, y_test = self.read_data('sir-toy-dataset/testNN.txt')
        # x_train, y_train = self.read_data('my-toy-dataset/trainNN.txt')
        # x_test, y_test = self.read_data('my-toy-dataset/testNN.txt')
        super().__init__(batch_size, x_train, y_train, x_test, y_test)


class CIFAR10Loader(DataLoader):
    @staticmethod
    def load_label_names():
        with open('cifar-10/batches.meta', 'rb') as f:
            dct = pickle.load(f, encoding='bytes')
        return dct[b'label_names']

    @staticmethod
    def read_data(data_file):
        with open(data_file, 'rb') as f:
            dct = pickle.load(f, encoding='bytes')
        x = dct[b'data']
        x = x.astype(float) / np.max(x)
        x = np.reshape(x, (len(x), 3, 32, 32))
        labels = dct[b'labels']
        y = one_hot_encode(labels)
        # y = np.expand_dims(y, axis=2)
        return x, y

    def draw_img(self, idx):
        print(self.label_names[np.argmax(self.y_train[idx])])
        img = self.x_train[idx].transpose((1, 2, 0))
        plt.imshow(img)
        plt.show()

    def __init__(self, batch_size):
        self.label_names = self.load_label_names()
        x_batches, y_batches = [], []
        for i in range(1, 6):
            x_batch, y_batch = self.read_data(f'cifar-10/data_batch_{i}')
            x_batches.append(x_batch)
            y_batches.append(y_batch)
        x_train = np.concatenate(x_batches)
        y_train = np.concatenate(y_batches)
        x_test, y_test = self.read_data('cifar-10/test_batch')
        super().__init__(batch_size, x_train, y_train, x_test, y_test)


class MNISTLoader(DataLoader):
    @staticmethod
    def read_data(img_file, label_file):
        x, y = loadlocal_mnist(images_path=img_file, labels_path=label_file)
        x = x.astype(float) / np.max(x)
        y = one_hot_encode(y)
        print(x.shape, y.shape)
        x = np.reshape(x, (len(x), 1, 28, 28))
        # y = np.expand_dims(y, axis=2)
        print(x.shape, y.shape)
        return x, y

    def draw_img(self, idx):
        print(np.argmax(self.y_train[idx]))
        img = np.squeeze(self.x_train[idx])
        plt.imshow(img, cmap='gray')
        plt.show()

    def __init__(self, batch_size):
        x_train, y_train = self.read_data('mnist/train-images.idx3-ubyte', 'mnist/train-labels.idx1-ubyte')
        x_test, y_test = self.read_data('mnist/t10k-images.idx3-ubyte', 'mnist/t10k-labels.idx1-ubyte')
        super().__init__(batch_size, x_train, y_train, x_test, y_test)


class Flatten:
    def __init__(self):
        self.x_shape = None

    def __repr__(self):
        return 'Flatten'

    def forward(self, x):
        self.x_shape = x.shape
        return np.reshape(x, (len(x), -1))

    def backward(self, dy):
        return np.reshape(dy, self.x_shape)


class Dense:
    def __init__(self, in_dim, out_dim, alpha=1e-3):
        self.weight = np.random.rand(in_dim, out_dim) * 0.001
        self.bias = np.random.rand(out_dim) * 0.001
        self.alpha = alpha
        self.x = None

    def __repr__(self):
        in_dim, out_dim = self.weight.shape
        return f'Dense {in_dim} -> {out_dim}'

    def forward(self, x):
        self.x = x
        return x @ self.weight + self.bias

    def backward(self, dy):
        dw = self.x.T @ dy
        db = np.average(dy, axis=0)
        self.weight -= self.alpha * dw
        self.bias -= self.alpha * db
        return dy @ self.weight.T


class ReLU:
    def __init__(self):
        self.x = None
        self.m = 0.001

    def __repr__(self):
        return 'ReLU'

    def forward(self, x):
        self.x = x
        return np.where(x < 0, 0, x * self.m)

    def backward(self, dy):
        return np.where(self.x < 0, 0, dy * self.m)


class Softmax:
    def __init__(self):
        self.y = None

    def __repr__(self):
        return 'Softmax'

    def forward(self, x):
        x -= np.max(x)
        y = np.exp(x)
        s = np.sum(y, axis=1, keepdims=True)
        s[s == 0] = 1
        y /= s
        self.y = y
        return y

    # dy = true labels
    def backward(self, dy):
        return self.y - dy


class Model:
    def __init__(self, arch_file, in_dim, alpha=1e-3):
        self.layers = []
        with open(arch_file) as f:
            for line in f:
                layer_data = line.split()
                layer_name = layer_data[0]
                if layer_name == 'FC':
                    out_dim = (int(layer_data[1]),)
                    self.layers.append(Dense(in_dim[0], out_dim[0], alpha))
                    in_dim = out_dim
                elif layer_name == 'ReLU':
                    self.layers.append(ReLU())
                elif layer_name == 'Softmax':
                    self.layers.append(Softmax())
                elif layer_name == 'Flatten':
                    self.layers.append(Flatten())
                    in_dim = (product(in_dim),)

    def __repr__(self):
        return '\n'.join(str(layer) for layer in self.layers)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, y):
        grad = y
        for layer in reversed(self.layers):
            grad = layer.backward(grad)


def calc_metrics(model, data):
    x, y = data
    y_pred = model.forward(x)
    loss = cross_entropy_loss(y, y_pred)
    acc = accuracy(y, y_pred)
    f1 = f1_score(y, y_pred)
    return loss, acc, f1


def train(model, dataloader, epochs=5):
    t_losses, v_losses = [], []
    for i in range(epochs):
        dataloader.reset()
        while dataloader.next():
            x, y = dataloader.next_train_batch()
            model.forward(x)
            model.backward(y)
        print(i)
        t_loss, t_acc, t_f1 = calc_metrics(model, dataloader.train_data())
        v_loss, v_acc, v_f1 = calc_metrics(model, dataloader.val_data())
        # print(f't_loss: {t_loss:>2.2f}, t_acc: {t_acc:>2.2f}, t_f1: {t_f1:>2.2f}')
        # print(f'v_loss: {v_loss:>2.2f}, v_acc: {v_acc:>2.2f}, v_f1: {v_f1:>2.2f}')
        t_losses.append(t_loss)
        v_losses.append(v_loss)
    t_loss, t_acc, t_f1 = calc_metrics(model, dataloader.train_data())
    v_loss, v_acc, v_f1 = calc_metrics(model, dataloader.val_data())
    print(f't_loss: {t_loss:>2.2f}, t_acc: {t_acc:>2.2f}, t_f1: {t_f1:>2.2f}')
    print(f'v_loss: {v_loss:>2.2f}, v_acc: {v_acc:>2.2f}, v_f1: {v_f1:>2.2f}')
    t_losses.append(t_loss)
    v_losses.append(v_loss)

    plt.plot(range(len(t_losses)), t_losses, color='red', lw=2)
    plt.plot(range(len(v_losses)), v_losses, color='blue', lw=2)
    plt.show()
    return {}


def log(arch_file, params, metrics):
    pass


def main():
    arch_file = 'input.txt'
    params = {
        'batch_size': 500,
        'epochs': 5,
        'alpha': 1e-2
    }

    # dataloader = ToyDataLoader(params['batch_size'])
    # dataloader = CIFAR10Loader(params['batch_size'])
    dataloader = MNISTLoader(params['batch_size'])
    # dataloader.draw_img(0)

    model = Model(arch_file, dataloader.shape(), params['alpha'])
    print(model)

    metrics = train(model, dataloader, params['epochs'])
    log(arch_file, params, metrics)


if __name__ == '__main__':
    main()
