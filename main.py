import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as sm
from sklearn.preprocessing import OneHotEncoder
import pickle
from mlxtend.data import loadlocal_mnist
import os

np.random.seed(4)
IMG_DIR = 'plots'


def safe_log(x):
    x[x <= 0] = 1
    return np.log(x)


def get_labels(y):
    return np.argmax(y, axis=0).ravel()


def cross_entropy_loss(y_true, y_pred):
    return -np.average(np.sum(y_true * safe_log(y_pred), axis=0))


def accuracy(y_true, y_pred):
    return sm.accuracy_score(get_labels(y_true), get_labels(y_pred))


def f1_score(y_true, y_pred):
    return sm.f1_score(get_labels(y_true), get_labels(y_pred), average='macro')


def one_hot_encode(a):
    return OneHotEncoder().fit_transform(np.reshape(a, (-1, 1))).toarray()


def shuffle_together(a, b):
    idx = np.random.permutation(len(a))
    return a[idx], b[idx]


class DataLoader:
    def __init__(self, batch_size, x_train, y_train, x_test, y_test):
        self.batch_size = batch_size
        self.cur = 0
        self.x_train = x_train
        self.y_train = y_train
        m = x_test.shape[-1] // 2
        self.x_val = x_test[..., :m]
        self.y_val = y_test[..., :m]
        self.x_test = x_test[..., m:]
        self.y_test = y_test[..., m:]

        print(f'x_train: {self.x_train.shape}, y_train: {self.y_train.shape}')
        print(f'x_val: {self.x_val.shape}, y_val: {self.y_val.shape}')
        print(f'x_test: {self.x_test.shape}, y_test: {self.y_test.shape}')

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
        labels = np.array(dct[b'labels'])
        x, labels = shuffle_together(x, labels)

        # taking only a subset of the labels
        # take = [i for i in range(len(labels)) if labels[i] in (0, 1)]
        # x = x[take]
        # labels = labels[take]

        x = x.astype(float) / np.max(x)
        x = np.reshape(x, (len(x), 3, 32, 32))
        x = x.transpose((2, 3, 1, 0))
        y = one_hot_encode(labels).T

        print(f'CIFAR-10: {x.shape}, {y.shape}')
        return x, y

    def draw_img(self, idx):
        label = self.label_names[np.argmax(self.y_train[:, idx])]
        img = self.x_train[..., idx]
        plt.imshow(img)
        plt.title(f'img-{idx} is {label}')
        plt.show()

    def __init__(self, batch_size):
        self.label_names = self.load_label_names()
        print(self.label_names)
        x_batches, y_batches = [], []
        for i in range(1, 6):
            x_batch, y_batch = self.read_data(f'cifar-10/data_batch_{i}')
            x_batches.append(x_batch)
            y_batches.append(y_batch)
        x_train = np.concatenate(x_batches, axis=-1)
        y_train = np.concatenate(y_batches, axis=-1)
        x_test, y_test = self.read_data('cifar-10/test_batch')
        # x_train, y_train = x_train[..., :500], y_train[:, :500]
        super().__init__(batch_size, x_train, y_train, x_test, y_test)


class MNISTLoader(DataLoader):
    @staticmethod
    def read_data(img_file, label_file):
        x, labels = loadlocal_mnist(images_path=img_file, labels_path=label_file)
        x, labels = shuffle_together(x, labels)

        # taking only a subset of the labels
        # take = [i for i in range(len(labels)) if labels[i] in (0, 1, 2)]
        # x = x[take]
        # labels = labels[take]

        x = x.astype(float) / np.max(x)
        x = np.reshape(x, (len(x), 1, 28, 28))
        x = x.transpose((2, 3, 1, 0))
        y = one_hot_encode(labels).T

        print(f'MNIST: {x.shape}, {y.shape}')
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
        # x_train, y_train = x_train[..., :500], y_train[:, :500]
        super().__init__(batch_size, x_train, y_train, x_test, y_test)


class Conv:
    def __init__(self, filter_count, filter_shape, stride=1, padding=0, alpha=1e-3):
        scale = np.sqrt(2.0 / np.prod(filter_shape))
        self.filter = np.random.randn(*filter_shape, filter_count) * scale
        self.bias = np.random.randn(filter_count, 1) * scale
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
        n, m, _, samples = x.shape
        a, b, _, f_cnt = self.filter.shape
        p = (n - a) // self.stride + 1
        q = (m - b) // self.stride + 1
        y = np.empty((p, q, f_cnt, samples))
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
        n, m, _, _ = self.x.shape
        a, b, _, _ = self.filter.shape
        p, q, _, _ = dy.shape
        df = np.empty(self.filter.shape)
        x = np.expand_dims(self.x, axis=3)
        dy = np.expand_dims(dy, axis=2)
        s = self.stride
        ps, qs = p * s, q * s
        for fi in range(a):
            for fj in range(b):
                df[fi, fj, :, :] = np.average(np.sum(
                    x[fi:(fi + ps):s, fj:(fj + qs):s, :, :, :] * dy,
                    axis=(0, 1)), axis=-1)

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

        self.filter -= self.alpha * df
        self.bias -= self.alpha * np.expand_dims(db, axis=-1)
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
        n, m, c, samples = x.shape
        a, b = self.shape
        p = (n - a) // self.stride + 1
        q = (m - b) // self.stride + 1
        y = np.empty((p, q, c, samples))

        l_dim = c * samples
        self.idx = np.empty((p, q, l_dim), dtype=int)

        for yi in range(p):
            i = yi * self.stride
            for yj in range(q):
                j = yj * self.stride
                slc = x[i:i + a, j:j + b, :, :]
                y[yi, yj, :, :] = np.max(slc, axis=(0, 1))
                self.idx[yi, yj, :] = np.argmax(slc.reshape(a * b, l_dim), axis=0)
        return y

    def backward(self, dy):
        p, q, _, _ = dy.shape
        n, m, c, samples = self.x_shape
        l_dim = c * samples
        use_idx = np.arange(l_dim)
        dy = np.reshape(dy, (p, q, -1))
        dx = np.zeros((n, m, l_dim))
        for yi in range(p):
            for yj in range(q):
                i, j = np.unravel_index(self.idx[yi, yj, :], self.shape)
                i += yi * self.stride
                j += yj * self.stride
                dx[i, j, use_idx] += dy[yi, yj, :]
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
        scale = np.sqrt(2.0 / in_dim)
        self.weight = np.random.randn(out_dim, in_dim) * scale
        self.bias = np.random.randn(out_dim, 1) * scale
        self.alpha = alpha
        self.x = None

    def __repr__(self):
        out_dim, in_dim = self.weight.shape
        return f'[Dense {in_dim} -> {out_dim}]'

    def forward(self, x):
        self.x = x
        return self.weight @ x + self.bias

    def backward(self, dy):
        dx = self.weight.T @ dy
        dw = dy @ self.x.T / dy.shape[1]
        db = np.expand_dims(np.average(dy, axis=-1), axis=-1)
        self.weight -= self.alpha * dw
        self.bias -= self.alpha * db
        return dx


class ReLU:
    def __init__(self):
        self.x = None

    def __repr__(self):
        return '[ReLU]'

    def forward(self, x):
        self.x = x
        return np.where(x < 0, 0, x)

    def backward(self, dy):
        return np.where(self.x < 0, 0, dy)


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
                    filter_shape = tuple((filter_dim, filter_dim, in_dim[2]))
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
                        in_dim = (np.prod(in_dim),)
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
                    in_dim = (np.prod(in_dim),)
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

    def plot(self, step_size):
        print(f'train {self.name}: {self.trains[-1]:.3f}')
        print(f'valid {self.name}: {self.vals[-1]:.3f}')
        cnt = len(self.trains)
        epochs = np.arange(cnt) * step_size
        extra = 'o' if cnt == 1 else ''
        plt.plot(epochs, self.trains, '-r' + extra, lw=2, label='train')
        plt.plot(epochs, self.vals, '-b' + extra, lw=2, label='val')
        plt.title(self.name)
        plt.legend()
        plt.savefig(os.path.join(IMG_DIR, f'{self.name}.png'))
        plt.close()


def train(model, dataloader, epochs=5):
    step_size = epochs // 20 if epochs > 20 else 1
    marks = range(epochs, -1, -step_size)
    p_loss, p_acc, p_f1 = Plotter('loss'), Plotter('accuracy'), Plotter('f1 score')

    for i in range(1, epochs + 1):
        y, y_pred = None, None
        metrics = np.zeros(3)
        cnt = 0

        dataloader.reset()
        while dataloader.next():
            x, y = dataloader.next_train_batch()
            y_pred = model.forward(x)
            model.backward(y)

            metrics += np.array(calc_metrics(y, y_pred))
            cnt += 1
            # print('.', end='', flush=True)

        if i in marks:
            print('#', end='', flush=True)
            # x, y = dataloader.train_data()
            # y_pred = model.forward(x)
            # t_loss, t_acc, t_f1 = calc_metrics(y, y_pred)
            t_loss, t_acc, t_f1 = metrics / cnt

            x_val, y_val = dataloader.val_data()
            y_pred = model.forward(x_val)
            v_loss, v_acc, v_f1 = calc_metrics(y_val, y_pred)

            print(i)
            print(f't = {t_loss:.3f} {t_acc:.3f} {t_f1:.3f}')
            print(f'v = {v_loss:.3f} {v_acc:.3f} {v_f1:.3f}')

            p_loss.add(t_loss, v_loss)
            p_acc.add(t_acc, v_acc)
            p_f1.add(t_f1, v_f1)
    print()

    p_loss.plot(step_size)
    p_acc.plot(step_size)
    p_f1.plot(step_size)

    # x_test, y_test = dataloader.test_data()
    # y_pred = model.forward(x_test)
    # print(calc_metrics(y_test, y_pred))

    return {}


def main():
    if not os.path.isdir(IMG_DIR):
        os.makedirs(IMG_DIR)

    arch_file = 'input.txt'
    params = {
        'batch_size': 32,
        'epochs': 5,
        'alpha': 1e-2
    }

    # dataloader = ToyDataLoader(params['batch_size'])
    # dataloader = CIFAR10Loader(params['batch_size'])
    dataloader = MNISTLoader(params['batch_size'])
    # dataloader.draw_img(0)

    model = Model(arch_file, dataloader.shape(), params['alpha'])
    print(model)

    model_out_dim = model.layers[-2].bias.shape[0]
    act_out_dim = dataloader.train_data()[1].shape[0]
    assert model_out_dim == act_out_dim, \
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
