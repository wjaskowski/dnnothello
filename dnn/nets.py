import sys
import logging
from os.path import join
import numpy as np

import caffe
from caffe import layers as L
from caffe import params as P
from util.redirector import Redirector

logger = logging.getLogger(__name__)


def cnn(source, batch_size, input_size=None, deploy=False):
    n = caffe.NetSpec()

    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=source, ntop=2)

    n.conv1 = L.Convolution(n.data, num_output=64, kernel_size=4, stride=1, pad=0, weight_filler=dict(type='xavier'),
                            bias_filler=dict(type='constant', value=0))
    n.relu1 = L.ReLU(n.conv1, in_place=True)
    n.conv2 = L.Convolution(n.relu1, num_output=64, kernel_size=3, stride=1, pad=1, weight_filler=dict(type='xavier'),
                            bias_filler=dict(type='constant', value=0))
    n.relu2 = L.ReLU(n.conv2, in_place=True)
    n.pool1 = L.Pooling(n.relu2, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    n.conv3 = L.Convolution(n.pool1, num_output=128, kernel_size=3, stride=1, pad=1, weight_filler=dict(type='xavier'),
                            bias_filler=dict(type='constant', value=0))
    n.relu3 = L.ReLU(n.conv3, in_place=True)
    n.conv4 = L.Convolution(n.relu3, num_output=128, kernel_size=3, stride=1, pad=1, weight_filler=dict(type='xavier'),
                            bias_filler=dict(type='constant', value=0))
    n.relu4 = L.ReLU(n.conv4, in_place=True)
    n.pool2 = L.Pooling(n.relu4, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    n.ip1 = L.InnerProduct(n.pool2, num_output=512, weight_filler=dict(type='gaussian', std=0.01),
                           bias_filler=dict(type='constant', value=0))
    n.relu5 = L.ReLU(n.ip1, in_place=True)
    n.dropout1 = L.Dropout(n.ip1, dropout_param=dict(dropout_ratio=0.5), in_place=True)

    n.ip2 = L.InnerProduct(n.ip1, num_output=512, weight_filler=dict(type='gaussian', std=0.01),
                           bias_filler=dict(type='constant', value=0))
    n.relu6 = L.ReLU(n.ip2, in_place=True)
    n.dropout2 = L.Dropout(n.ip2, dropout_param=dict(dropout_ratio=0.5), in_place=True)

    n.ip3 = L.InnerProduct(n.ip2, num_output=60, weight_filler=dict(type='gaussian', std=0.01),
                           bias_filler=dict(type='constant', value=0))
    n.acc = L.Accuracy(n.ip3, n.label, include=dict(phase=1))

    if not deploy:
        n.loss = L.SoftmaxWithLoss(n.ip3, n.label)
        return n.to_proto()
    else:
        assert input_size is not None
        deploy_str = 'input: {}\n' \
                     'input_dim: {}\n' \
                     'input_dim: {}\n' \
                     'input_dim: {}\n' \
                     'input_dim: {}'.format('"data"', batch_size, 3, input_size, input_size)
        n.prob = L.Softmax(n.ip3)
        return deploy_str + '\n' + 'layer {' + 'layer {'.join(str(n.to_proto()).split('layer {')[2:])


def cnn_nopool(source, batch_size, input_size=None, deploy=False):
    n = caffe.NetSpec()

    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=source, ntop=2)

    n.conv1 = L.Convolution(n.data, num_output=64, kernel_size=3, stride=1, pad=1, weight_filler=dict(type='xavier'),
                            bias_filler=dict(type='constant', value=0))
    n.relu1 = L.ReLU(n.conv1, in_place=True)
    n.conv2 = L.Convolution(n.relu1, num_output=64, kernel_size=3, stride=1, pad=1, weight_filler=dict(type='xavier'),
                            bias_filler=dict(type='constant', value=0))
    n.relu2 = L.ReLU(n.conv2, in_place=True)

    n.conv3 = L.Convolution(n.relu2, num_output=128, kernel_size=3, stride=1, pad=1, weight_filler=dict(type='xavier'),
                            bias_filler=dict(type='constant', value=0))
    n.relu3 = L.ReLU(n.conv3, in_place=True)
    n.conv4 = L.Convolution(n.relu3, num_output=128, kernel_size=3, stride=1, pad=1, weight_filler=dict(type='xavier'),
                            bias_filler=dict(type='constant', value=0))
    n.relu4 = L.ReLU(n.conv4, in_place=True)

    n.conv5 = L.Convolution(n.relu4, num_output=256, kernel_size=3, stride=1, pad=1, weight_filler=dict(type='xavier'),
                            bias_filler=dict(type='constant', value=0))
    n.relu5 = L.ReLU(n.conv5, in_place=True)
    n.conv6 = L.Convolution(n.relu5, num_output=256, kernel_size=3, stride=1, pad=1, weight_filler=dict(type='xavier'),
                            bias_filler=dict(type='constant', value=0))
    n.relu6 = L.ReLU(n.conv6, in_place=True)

    n.ip1 = L.InnerProduct(n.relu6, num_output=128, weight_filler=dict(type='gaussian', std=0.01),
                           bias_filler=dict(type='constant', value=0))
    n.relu7 = L.ReLU(n.ip1, in_place=True)
    # n.dropout1 = L.Dropout(n.ip1, dropout_param=dict(dropout_ratio=0.5), in_place=True)

    # n.ip2 = L.InnerProduct(n.ip1, num_output=128, weight_filler=dict(type='gaussian', std=0.01),
    #                        bias_filler=dict(type='constant', value=0))
    # n.relu6 = L.ReLU(n.ip2, in_place=True)
    # n.dropout2 = L.Dropout(n.ip2, dropout_param=dict(dropout_ratio=0.5), in_place=True)

    n.ip3 = L.InnerProduct(n.ip1, num_output=60, weight_filler=dict(type='gaussian', std=0.01),
                           bias_filler=dict(type='constant', value=0))
    n.acc = L.Accuracy(n.ip3, n.label, include=dict(phase=1))

    if not deploy:
        n.loss = L.SoftmaxWithLoss(n.ip3, n.label)
        return n.to_proto()
    else:
        assert input_size is not None
        deploy_str = 'input: {}\n' \
                     'input_dim: {}\n' \
                     'input_dim: {}\n' \
                     'input_dim: {}\n' \
                     'input_dim: {}'.format('"data"', batch_size, 3, input_size, input_size)
        n.prob = L.Softmax(n.ip3)
        return deploy_str + '\n' + 'layer {' + 'layer {'.join(str(n.to_proto()).split('layer {')[2:])


def convert_proto_to_deploy(proto_file, batch_size, channels, patch_size):
    with open(proto_file, 'r') as f:
        train_proto = f.read()
    deploy_str = 'input: {}\n' \
                 'input_dim: {}\n' \
                 'input_dim: {}\n' \
                 'input_dim: {}\n' \
                 'input_dim: {}'.format('"data"', batch_size, channels, patch_size, patch_size)
    softmax_str = 'layer {\n ' \
                  'name: "prob"\n ' \
                  'type: "Softmax"\n ' \
                  'bottom: "ip3"\n ' \
                  'top: "prob"\n }\n'
    return deploy_str + '\n' + 'layer {' + 'layer {'.join(train_proto.split('layer {')[2:-2]) + '\n' + softmax_str


def create_sgd_solver(out, train_net, test_net, iters=30000, snapshot_prefix='model', base_lr=0.1, gamma=0.33,
                      step_size=10000):
    def decorate_with_quotation(text):
        return '\"' + text + '\"'

    config = {
        'train_net': decorate_with_quotation(train_net),
        'test_net': decorate_with_quotation(test_net),
        'test_iter': '400',
        'test_interval': '625',
        'test_initialization': 'false',

        'base_lr': base_lr,
        'momentum': '0.9',
        'weight_decay': '0.0005',

        'lr_policy': '\"step\"',
        'gamma': gamma,
        'stepsize': step_size,

        'display': '200',
        'max_iter': iters,

        'snapshot': '5000',
        'snapshot_prefix': decorate_with_quotation(snapshot_prefix),
        'solver_mode': 'GPU'
    }
    with open(out, 'w') as fp:
        fp.writelines([k + ': ' + str(v) + '\n' for k, v in config.iteritems()])


def create_rmsprop_solver(out, train_net, test_net, iters=30000, snapshot_prefix='model', base_lr=0.001, gamma=0.0001,
                          step_size=10000):
    def decorate_with_quotation(text):
        return '\"' + text + '\"'

    config = {
        'train_net': decorate_with_quotation(train_net),
        'test_net': decorate_with_quotation(test_net),
        'test_iter': '400',
        'test_interval': '625',
        'test_initialization': 'false',

        'base_lr': base_lr,
        'momentum': '0',
        'weight_decay': '0.0005',

        'lr_policy': '\"inv\"',
        'gamma': 0.0001,
        'power': 0.75,

        'display': '200',
        'max_iter': iters,

        'snapshot': '5000',
        'snapshot_prefix': decorate_with_quotation(snapshot_prefix),
        'solver_mode': 'GPU',
        'solver_type': 'RMSPROP',
        'rms_decay': 0.98
    }
    with open(out, 'w') as fp:
        fp.writelines([k + ': ' + str(v) + '\n' for k, v in config.iteritems()])


def deploy_model(model_dir, data_dir, model_type='cnn', solver_type='sgd', train_batch=256, test_batch=100, iters=30000,
                 prefix='model', lr=0.001, gamma=0.1, step=10000):
    with open(join(model_dir, 'train.prototxt'), 'w') as f:
        f.write(str(getattr(sys.modules[__name__], model_type)(join(data_dir, 'train'), train_batch)))

    with open(join(model_dir, 'test.prototxt'), 'w') as f:
        f.write(str(getattr(sys.modules[__name__], model_type)(join(data_dir, 'test'), test_batch)))

    solver_creator = create_sgd_solver
    if solver_type == 'sgd':
        print 'Using SGD'
        solver_creator = create_sgd_solver
    elif solver_type == 'rmsprop':
        print 'Using RMSProp'
        solver_creator = create_rmsprop_solver

    solver_creator(join(model_dir, 'solver.prototxt'),
                   join(model_dir, 'train.prototxt'),
                   join(model_dir, 'test.prototxt'),
                   iters,
                   join(model_dir, prefix),
                   lr,
                   gamma,
                   step)


def load_net(model_dir, model_name, input_size, batch_size, channels, model_iter, gpu):
    # deploy the trained net
    deploy_file = join(model_dir, 'deploy.prototxt')
    with open(deploy_file, 'w') as f:
        f.write(convert_proto_to_deploy(join(model_dir, 'train.prototxt'), batch_size, channels, input_size))

    # instantiate the net
    net_file = join(model_dir, '{}_iter_{}.caffemodel'.format(model_name, model_iter))
    caffe.set_device(gpu)
    caffe.set_mode_gpu()

    with Redirector():
        net = caffe.Net(deploy_file, net_file, caffe.TEST)
    return net


def get_solver(model_dir, device=1):
    caffe.set_device(device)
    caffe.set_mode_gpu()
    solver = caffe.get_solver(join(model_dir, 'solver.prototxt'))
    return solver


def predict(net, examples):
    net.blobs['data'].data[...] = np.asarray(examples)
    net.forward(start='conv1')
    predictions = (np.exp(net.blobs['ip3'].data).T / np.exp(net.blobs['ip3'].data).sum(1).T).T
    return predictions
