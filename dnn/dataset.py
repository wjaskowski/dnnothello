import numpy as np
import os
from os.path import join

from sklearn.cross_validation import train_test_split
from sklearn import preprocessing

from util.io import LMDBReader, save_lmdb
from dnn.nets import deploy_model

__author__ = 'pliskowski'


def split_train_test(experiment_dir, data_path, model_type, train_batch, test_batch, iters, test_size=0.33, seed=123):
    model_dir = join(experiment_dir, model_type)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    dataset_dir = join(model_dir, 'dataset')
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    reader = LMDBReader(data_path)
    data, labels = reader.read_all()
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    labels = le.transform(labels)

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=seed)

    save_lmdb(join(dataset_dir, 'train'), x_train, y_train)
    save_lmdb(join(dataset_dir, 'test'), x_test, y_test)

    deploy_model(model_dir, dataset_dir, model_type, train_batch, test_batch, iters, model_type)


def get_num_unique_labels():
    reader = LMDBReader('../games/train')
    data, labels = reader.read_all()

    print np.unique(labels)
    print len(np.unique(labels))

if __name__ == '__main__':
    get_num_unique_labels()
