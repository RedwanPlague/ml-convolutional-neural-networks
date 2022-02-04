import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as sm
import pickle
from mlxtend.data import loadlocal_mnist
import os

np.random.seed(4)
IMG_DIR = 'plots'
DEC = 1e-3


def safe_log(x):
    x[x <= 0] = 1
    return np.log(x)


# returns product of all elements
def product(a):
    prod = 1
    for x in a:
        prod *= x
    return prod


def get_labels(y):
    return np.argmax(y, axis=0).ravel()


def cross_entropy_loss(y_true, y_pred):
    return -np.average(np.sum(y_true * safe_log(y_pred), axis=0))


def accuracy(y_true, y_pred):
    return sm.accuracy_score(get_labels(y_true), get_labels(y_pred))


def f1_score(y_true, y_pred):
    return sm.f1_score(get_labels(y_true), get_labels(y_pred), average='macro')


# list to one hot encoded numpy array
def one_hot_encode(a):
    b = np.zeros((len(a), max(a) + 1))
    b[range(len(a)), a] = 1
    return b


def scale(a):
    return (2 * a - 1) * DEC


class DataLoader:
    def __init__(self, batch_size, x_train, y_train, x_test, y_test):
        self.batch_size = batch_size
        self.cur = 0
        self.x_train = x_train
        self.y_train = y_train
        m = len(x_test) // 2
        # m = x_test.shape[-1]
        self.x_val = x_test[..., :m]
        self.y_val = y_test[..., :m]
        self.x_test = x_test[..., m:]
        self.y_test = y_test[..., m:]

    def shape(self):
        return self.x_train.shape[:-1]

    def reset(self):
        self.cur = 0

    def next(self):
        return self.cur < self.x_train.shape[-1]

    def next_train_batch(self):
        end = self.cur + self.batch_size
        x_train_batch = self.x_train[..., self.cur: end]
        y_train_batch = self.y_train[..., self.cur: end]
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
        x = x.astype(float).T / np.max(x)
        y = np.array(df.iloc[:, -1])
        y = np.array(pd.get_dummies(y)).T
        print(x.shape, y.shape)
        return x, y

    def __init__(self, batch_size):
        # x_train, y_train = self.read_data('sir-toy-dataset/trainNN.txt')
        # x_test, y_test = self.read_data('sir-toy-dataset/testNN.txt')
        x_train, y_train = self.read_data('my-toy-dataset/trainNN.txt')
        x_test, y_test = self.read_data('my-toy-dataset/testNN.txt')
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
        labels = dct[b'labels']

        # taking only a subset of the labels
        take = [i for i in range(len(labels)) if labels[i] in (0, 1)]
        x = x[take]
        labels = labels[take]

        x = x.astype(float) / np.max(x)
        x = np.reshape(x, (len(x), 3, 32, 32))
        x = x.transpose((2, 3, 1, 0))
        y = one_hot_encode(labels).T
        return x, y

    def draw_img(self, idx):
        label = self.label_names[np.argmax(self.y_train[:, idx])]
        img = self.x_train[..., idx]
        plt.imshow(img)
        plt.title(f'img-{idx} is {label}')
        plt.show()

    def __init__(self, batch_size):
        self.label_names = self.load_label_names()
        x_batches, y_batches = [], []
        for i in range(1, 6):
            x_batch, y_batch = self.read_data(f'cifar-10/data_batch_{i}')
            x_batches.append(x_batch)
            y_batches.append(y_batch)
        x_train = np.concatenate(x_batches, axis=-1)
        y_train = np.concatenate(y_batches, axis=-1)
        x_test, y_test = self.read_data('cifar-10/test_batch')
        super().__init__(batch_size, x_train, y_train, x_test, y_test)


class MNISTLoader(DataLoader):
    @staticmethod
    def read_data(img_file, label_file):
        x, labels = loadlocal_mnist(images_path=img_file, labels_path=label_file)

        # taking only a subset of the labels
        take = [i for i in range(len(labels)) if labels[i] in (0, 1)]
        x = x[take]
        labels = labels[take]

        x = x.astype(float) / np.max(x)
        x = np.reshape(x, (len(x), 1, 28, 28))
        x = x.transpose((2, 3, 1, 0))
        y = one_hot_encode(labels).T
        print(x.shape, y.shape)
        return x, y

    def draw_img(self, idx):
        label = np.argmax(self.y_train[:, idx])
        img = np.squeeze(self.x_train[..., idx])
        plt.imshow(img, cmap='gray')
        plt.title(f'img-{idx} is {label}')
        plt.show()

    def __init__(self, batch_size):
        x_train, y_train = self.read_data('mnist/train-images.idx3-ubyte', 'mnist/train-labels.idx1-ubyte')
        x_test, y_test = self.read_data('mnist/t10k-images.idx3-ubyte', 'mnist/t10k-labels.idx1-ubyte')
        super().__init__(batch_size, x_train, y_train, x_test, y_test)


class Conv:
    def __init__(self, filter_count, filter_shape, stride=1, padding=0, alpha=1e-3):
        self.filter = np.random.rand(*filter_shape, filter_count)
        self.bias = np.random.rand(filter_count, 1)
        self.stride = stride
        self.padding = padding
        self.alpha = alpha
        self.x_act_shape = None
        self.x = None

    def __repr__(self):
        a, b, c, k = self.filter.shape
        return f'[Convolution {k} x {a}x{b}x{c} s{self.stride} p{self.padding}]'

    def forward(self, x):
        self.x_act_shape = x.shape
        x = np.pad(x, ((self.padding,), (self.padding,), (0,), (0,)), constant_values=0)
        self.x = x
        n, m, c, _ = x.shape
        a, b, _, f_cnt = self.filter.shape
        p = (n - a) // self.stride + 1
        q = (m - b) // self.stride + 1
        y = np.empty((p, q, f_cnt, x.shape[-1]))
        x = np.expand_dims(x, axis=-2)
        f = np.expand_dims(self.filter, axis=-1)
        i = 0
        for yi in range(p):
            j = 0
            for yj in range(q):
                y[yi, yj, :, :] = np.sum(x[i:i + a, j:j + b, :, :, :] * f, axis=(0, 1, 2)) + self.bias
                j += self.stride
            i += self.stride
        return y

    def calc_pos(self, i, n, a):
        s = self.stride
        w = a - 1
        i += self.padding
        d = i % s
        i -= d
        w -= d
        i //= s
        while i + a > n and w >= 0 and i >= 0:
            i -= 1
            w -= s
        if i < 0 or w < 0:
            return [-1] * 4
        length = min(w // s, i)
        tw = w - length * s
        ti = i - length
        return tw, w, ti, i

    def backward(self, dy):
        db = np.average(np.sum(dy, axis=(0, 1)), axis=-1)
        self.bias -= self.alpha * np.expand_dims(db, axis=-1)
        n, m, c, _ = self.x.shape
        a, b, f_cnt, _ = self.filter.shape
        p, q, _, _ = dy.shape
        df = np.empty(self.filter.shape)
        x = np.expand_dims(self.x, axis=3)
        dy = np.expand_dims(dy, axis=2)
        s = self.stride
        ps, qs = p * s, q * s
        for fi in range(a):
            for fj in range(b):
                df[fi, fj, :, :] = np.average(np.sum(
                    x[fi:fi + ps:s, fj:fj + qs:s, :, :, :] * dy,
                    axis=(0, 1)), axis=-1)
        self.filter -= self.alpha * df

        an, am, _, _ = self.x_act_shape
        dx = np.zeros(self.x_act_shape)
        f = np.expand_dims(self.filter, axis=-1)
        f = np.flip(f, axis=(0, 1))

        for i in range(an):
            for j in range(am):
                twi, wi, tyi, yi = self.calc_pos(i, n, a)
                twj, wj, tyj, yj = self.calc_pos(j, m, b)
                f_slc = f[twi:(wi + 1):s, twj:(wj + 1):s, :, :, :]
                dy_slc = dy[tyi:(yi + 1), tyj:(yj + 1), :, :, :]
                dx[i, j, :, :] += np.sum(f_slc * dy_slc, axis=(0, 1, 3))
        return dx


class Pool:
    def __init__(self, shape, stride=1):
        self.shape = shape
        self.stride = stride
        self.x_shape = None
        self.idx = None

    def __repr__(self):
        a, b = self.shape
        return f'[Max Pool {a}x{b} s{self.stride}]'

    def forward(self, x):
        self.x_shape = x.shape
        n, m, c, points = x.shape
        a, b = self.shape
        p = (n - a) // self.stride + 1
        q = (m - b) // self.stride + 1
        y = np.empty((p, q, c, points))

        d_lim = c * points
        self.idx = np.empty((p, q, d_lim), dtype=int)

        for yi in range(p):
            i = yi * self.stride
            for yj in range(q):
                j = yj * self.stride
                slc = x[i:i + a, j:j + b, :, :]
                y[yi, yj, :, :] = np.max(slc, axis=(0, 1))
                self.idx[yi, yj, :] = np.argmax(slc.reshape(a * b, d_lim), axis=0)
        return y

    def backward(self, dy):
        p, q, _, _ = dy.shape
        n, m, c, points = self.x_shape
        l_dim = c * points
        dy = np.reshape(dy, (p, q, -1))
        dx = np.zeros((n, m, l_dim))
        for yi in range(p):
            for yj in range(q):
                i, j = np.unravel_index(self.idx[yi, yj, :], self.shape)
                i += yi * self.stride
                j += yj * self.stride
                dx[i, j, np.arange(l_dim)] += dy[yi, yj, :]
        return dx.reshape(self.x_shape)


class Flatten:
    def __init__(self):
        self.x_shape = None

    def __repr__(self):
        return '[Flatten]'

    def forward(self, x):
        self.x_shape = x.shape
        return np.reshape(x, (-1, x.shape[-1]))

    def backward(self, dy):
        return np.reshape(dy, self.x_shape)


class Dense:
    def __init__(self, in_dim, out_dim, alpha=1e-3):
        self.weight = scale(np.random.rand(out_dim, in_dim))
        self.bias = scale(np.random.rand(out_dim, 1))
        self.alpha = alpha
        self.x = None

    def __repr__(self):
        out_dim, in_dim = self.weight.shape
        return f'[Dense {in_dim} -> {out_dim}]'

    def forward(self, x):
        self.x = x
        return self.weight @ x + self.bias

    def backward(self, dy):
        dw = dy @ self.x.T
        db = np.expand_dims(np.average(dy, axis=-1), axis=-1)
        self.weight -= self.alpha * dw
        self.bias -= self.alpha * db
        return self.weight.T @ dy


class ReLU:
    def __init__(self):
        self.x = None
        self.m = DEC

    def __repr__(self):
        return '[ReLU]'

    def forward(self, x):
        self.x = x
        return np.where(x < 0, 0, x * self.m)

    def backward(self, dy):
        return np.where(self.x < 0, 0, dy * self.m)


class Softmax:
    def __init__(self):
        self.y = None

    def __repr__(self):
        return '[Softmax]'

    def forward(self, x):
        x -= np.max(x, axis=0)
        y = np.exp(x)
        s = np.sum(y, axis=0, keepdims=True)
        s[s == 0] = 1
        y /= s
        self.y = y
        return y

    # dy = true labels
    def backward(self, dy):
        return self.y - dy


class Model:
    def __init__(self, arch_file, in_dim, alpha=1e-3):
        print(f'{in_dim} <- Input')
        self.layers = []
        with open(arch_file) as f:
            for line in f:
                layer_data = line.split()
                layer_name = layer_data[0]
                if layer_name == 'Conv':
                    filter_count = int(layer_data[1])
                    filter_dim = int(layer_data[2])
                    stride = int(layer_data[3])
                    padding = int(layer_data[4])
                    filter_shape = (filter_dim, filter_dim, in_dim[2])
                    self.layers.append(Conv(filter_count, filter_shape, stride, padding))
                    in_dim = (
                        (in_dim[0] + 2 * padding - filter_shape[0]) // stride + 1,
                        (in_dim[1] + 2 * padding - filter_shape[1]) // stride + 1,
                        filter_count
                    )
                elif layer_name == 'Pool':
                    filter_dim = int(layer_data[1])
                    stride = int(layer_data[2])
                    filter_shape = (filter_dim, filter_dim)
                    self.layers.append(Pool(filter_shape, stride))
                    in_dim = (
                        (in_dim[0] - filter_shape[0]) // stride + 1,
                        (in_dim[1] - filter_shape[1]) // stride + 1,
                        in_dim[2]
                    )
                elif layer_name == 'FC':
                    if len(in_dim) > 1:
                        self.layers.append(Flatten())
                        in_dim = (product(in_dim),)
                        print(f'{in_dim} <- {self.layers[-1]}')
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
                print(f'{in_dim} <- {self.layers[-1]}')

    def __repr__(self):
        return '-\n' + '\n'.join(str(layer) for layer in self.layers) + '\n-'

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, y):
        grad = y
        for layer in reversed(self.layers):
            grad = layer.backward(grad)


def calc_metrics(y_act, y_pred):
    loss = cross_entropy_loss(y_act, y_pred)
    acc = accuracy(y_act, y_pred)
    f1 = f1_score(y_act, y_pred)
    return loss, acc, f1


class Plotter:
    def __init__(self, name):
        self.name = name
        self.trains = []
        self.vals = []

    def add(self, t, v):
        self.trains.append(t)
        self.vals.append(v)

    def plot(self):
        cnt = len(self.trains)
        extra = 'o' if cnt == 1 else ''
        plt.plot(range(cnt), self.trains, '-r' + extra, lw=2)
        plt.plot(range(cnt), self.vals, '-b' + extra, lw=2)
        plt.title(self.name)
        plt.savefig(os.path.join(IMG_DIR, f'{self.name}.png'))
        plt.close()


def train(model, dataloader, epochs=5):
    step_size = epochs // 20 if epochs > 20 else 1
    marks = range(epochs - 1, -1, -step_size)
    p_loss, p_acc, p_f1 = Plotter('loss'), Plotter('accuracy'), Plotter('f1 score')

    for i in range(epochs):
        y, y_pred = None, None

        dataloader.reset()
        while dataloader.next():
            x, y = dataloader.next_train_batch()
            y_pred = model.forward(x)
            model.backward(y)
            print('.', end='', flush=True)

        if i in marks:
            print('#', end='', flush=True)
            t_loss, t_acc, t_f1 = calc_metrics(y, y_pred)

            x_val, y_val = dataloader.val_data()
            y_pred = model.forward(x_val)
            v_loss, v_acc, v_f1 = calc_metrics(y_val, y_pred)

            p_loss.add(t_loss, v_loss)
            p_acc.add(t_acc, v_acc)
            p_f1.add(t_f1, v_f1)

            print(i)
    print()

    p_loss.plot()
    p_acc.plot()
    p_f1.plot()

    return {}


def main():
    if not os.path.isdir(IMG_DIR):
        os.makedirs(IMG_DIR)

    arch_file = 'input.txt'
    params = {
        'batch_size': 500,
        'epochs': 5,
        'alpha': 1e-2
    }

    # x = np.random.rand(32, 32, 3, 50)
    # y = np.random.rand(10, 50)
    # model = Model(arch_file, x.shape[:3], params['alpha'])
    # model.forward(x)
    # model.backward(y)

    # x = np.random.rand(28, 28, 3, 32)
    # y = None

    # n = 10

    # conv = Conv(3, (3, 3, 3), 1, 1)
    # b = time.time()
    # for _ in range(n):
    #     y = conv.forward(x)
    # print(f'time = {(time.time() - b) / n * 1e3:.3f}ms')

    # b = time.time()
    # for _ in range(n):
    #     conv.backward(y)
    # print(f'time = {(time.time() - b) / n * 1e3:.3f}ms')

    # pool = Pool((1, 1), 1)
    # b = time.time()
    # for _ in range(n):
    #     y = pool.forward(x)
    # print(f'time = {(time.time() - b) / n * 1e3:.3f}ms')

    # b = time.time()
    # for _ in range(n):
    #     pool.backward(y)
    # print(f'time = {(time.time() - b) / n * 1e3:.3f}ms')

    # dataloader = ToyDataLoader(params['batch_size'])
    # dataloader = CIFAR10Loader(params['batch_size'])
    dataloader = MNISTLoader(params['batch_size'])
    # dataloader.draw_img(0)

    model = Model(arch_file, dataloader.shape(), params['alpha'])
    print(model)

    model_out_dim = model.layers[-2].bias.shape[0]
    act_out_dim = dataloader.train_data()[1].shape[0]
    assert model_out_dim == act_out_dim,\
        f'model gives {model_out_dim} labels, but data has {act_out_dim}'

    train(model, dataloader, params['epochs'])

    # x, y = dataloader.train_data()
    # x, y = x[..., :10], y[:, :10]
    # y_pred = model.forward(x)
    # act = get_labels(y)
    # pred = get_labels(y_pred)
    # print(act)
    # print(pred)
    # for i in range(5):
    #     dataloader.draw_img(i)


if __name__ == '__main__':
    main()
