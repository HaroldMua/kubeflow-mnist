from tensorflow import keras
import argparse
import os
import pickle
import numpy as np
import gzip

# 定义加载数据的函数，data_folder为保存gz数据的文件夹，该文件夹下有4个文件
# 'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
# 't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
def load_data(data_folder):

  files = [
      'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
      't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
  ]

  paths = []
  for fname in files:
    paths.append(os.path.join(data_folder,fname))

  with gzip.open(paths[0], 'rb') as lbpath:
    y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

  with gzip.open(paths[1], 'rb') as imgpath:
    x_train = np.frombuffer(
        imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

  with gzip.open(paths[2], 'rb') as lbpath:
    y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

  with gzip.open(paths[3], 'rb') as imgpath:
    x_test = np.frombuffer(
        imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

  return (x_train, y_train), (x_test, y_test)


def preprocess(data_dir: str):
	
    # fashion_mnist = keras.datasets.fashion_mnist
    # (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    (train_images, train_labels), (test_images, test_labels) = load_data('{pv_dir}/fashion-mnist'.format(pv_dir=data_dir))

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    os.makedirs(data_dir, exist_ok=True)

    with open(os.path.join(data_dir, 'train_images.pickle'), 'wb') as f:
        pickle.dump(train_images, f)

    with open(os.path.join(data_dir, 'train_labels.pickle'), 'wb') as f:
        pickle.dump(train_labels, f)

    with open(os.path.join(data_dir, 'test_images.pickle'), 'wb') as f:
        pickle.dump(test_images, f)

    with open(os.path.join(data_dir, 'test_labels.pickle'), 'wb') as f:
        pickle.dump(test_labels, f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Kubeflow MNIST training script')
    parser.add_argument('--data_dir', help='path to images and labels.')
    args = parser.parse_args()

    preprocess(data_dir=args.data_dir)
