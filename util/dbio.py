import caffe
import lmdb
import numpy as np

from caffe.proto import caffe_pb2
from itertools import islice
from progressbar import ProgressBar, Counter, Bar, Percentage, ETA


__author__ = 'pliskowski'


def save_lmdb(db_file, data, labels):
    assert data.shape[0] == labels.shape[0]

    widgets = ['Building LMDB: ', Counter(), ' ', Bar(), ' ', Percentage(), ' ', ETA()]
    bar = ProgressBar(widgets=widgets, maxval=len(labels)).start()

    in_db = lmdb.open(db_file, map_size=int(1e12))
    with in_db.begin(write=True) as in_txn:
        for in_idx, label in enumerate(labels):
            im_dat = caffe.io.array_to_datum(data[in_idx], label.astype(int))
            in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
            bar.update(bar.currval + 1)
    in_db.close()


class LMDBStreamer(object):
    def __init__(self, file_name, batches=None, msg='Streaming LMDB batches: '):
        self.fname = file_name
        self.pointer = 0
        self.msg = msg
        self.batches = batches

    def __enter__(self):
        widgets = [self.msg, Counter(), ' ', Bar(), ' ', Percentage(), ' ', ETA()]
        self.bar = ProgressBar(widgets=widgets, maxval=self.batches).start()
        return self

    def __exit__(self):
        self.bar.finish()

    def write(self, data, label, progress=True):
        db = lmdb.open(self.fname, map_size=int(1e12))
        with db.begin(write=True) as txn:
            for in_idx, label in enumerate(label):
                im_dat = caffe.io.array_to_datum(data[in_idx], label.astype(np.int))
                txn.put('{:0>10d}'.format(in_idx + self.pointer), im_dat.SerializeToString(), append=True)
        self.pointer += data.shape[0]
        if progress:
            self.bar.update(self.bar.currval + 1)
        db.close()


class LMDBReader(object):
    def __init__(self, file_name):
        self.file_name = file_name

    def read(self):
        lmdb_env = lmdb.open(self.file_name)
        lmdb_txn = lmdb_env.begin()
        lmdb_cursor = lmdb_txn.cursor()
        datum = caffe_pb2.Datum()

        for key, value in lmdb_cursor:
            datum.ParseFromString(value)
            label = datum.label
            data = caffe.io.datum_to_array(datum)
            yield data, label

        lmdb_env.close()

    def read_batches(self, batch_size, pad=True):
        shape = self.read_first().shape
        lmdb_env = lmdb.open(self.file_name)
        num_items = lmdb_env.stat()['entries']

        lmdb_txn = lmdb_env.begin()
        lmdb_cursor = lmdb_txn.cursor()
        datum = caffe_pb2.Datum()

        remainder = num_items % batch_size
        num_batches = num_items / batch_size

        c = lmdb_cursor.iternext()

        # yield full batches
        for b in range(num_batches):
            data = np.zeros((batch_size,) + shape, dtype=np.uint8)
            labels = np.zeros(batch_size, dtype=np.uint8)

            # read batch_size items from lmdb
            for i, (_, value) in enumerate(islice(c, batch_size)):
                datum.ParseFromString(value)
                labels[i] = datum.label
                data[i] = caffe.io.datum_to_array(datum)
            yield data, labels

        # yield last padded batch, if any
        if remainder > 0:
            if pad:
                data = np.zeros((batch_size,) + shape)
                labels = np.zeros(batch_size)
            else:
                data = np.zeros((remainder,) + shape)
                labels = np.zeros(remainder)

            num_read = 0
            for i, (_, value) in enumerate(c):
                datum.ParseFromString(value)
                labels[i] = datum.label
                data[i] = caffe.io.datum_to_array(datum)
                num_read += 1
            assert num_read == remainder
            yield data, labels

        lmdb_env.close()

    def read_all(self):
        lmdb_env = lmdb.open(self.file_name)
        num_items = lmdb_env.stat()['entries']
        data = np.zeros((num_items,) + self.read_first().shape, dtype=np.uint8)
        labels = np.zeros(num_items, dtype=np.uint8)

        lmdb_txn = lmdb_env.begin()
        lmdb_cursor = lmdb_txn.cursor()
        datum = caffe_pb2.Datum()

        widgets = ['Reading LMDB database: ', Counter(), ' ', Bar(), ' ', Percentage(), ' ', ETA()]
        bar = ProgressBar(widgets=widgets, maxval=num_items).start()
        for i, (key, value) in enumerate(lmdb_cursor):
            datum.ParseFromString(value)
            labels[i] = datum.label
            data[i] = caffe.io.datum_to_array(datum)
            bar.update(bar.currval + 1)
        bar.finish()

        lmdb_env.close()
        return data, labels

    def read_first(self):
        lmdb_env = lmdb.open(self.file_name)
        lmdb_txn = lmdb_env.begin()
        lmdb_cursor = lmdb_txn.cursor()
        datum = caffe_pb2.Datum()

        _, v = lmdb_cursor.iternext().next()
        datum.ParseFromString(v)
        data = caffe.io.datum_to_array(datum)
        lmdb_env.close()

        return data

    def size(self):
        lmdb_env = lmdb.open(self.file_name)
        num_items = lmdb_env.stat()['entries']
        lmdb_env.close()
        return num_items