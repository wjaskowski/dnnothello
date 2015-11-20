from __future__ import division
from itertools import izip_longest
import numpy as np
import os
import math
import matplotlib
import matplotlib.pyplot as plot
import matplotlib.cm as cm
from matplotlib.ticker import FuncFormatter
from matplotlib import rc
import pandas as pd
import seaborn as sns

pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

matplotlib.use('Agg')

rc('text', usetex=True)
rc('font', family='serif', serif=['Times'], variant='small-caps', size=6)
rc('legend', fontsize=6)

COLUMN_WIDTH_PT = 252.0
PT_PER_INCH = 72
DEFAULT_PLOT_SIZE = COLUMN_WIDTH_PT / PT_PER_INCH


def jet_colors(n):
    colors = []
    for i in range(1, n+1):
        colors.append(matplotlib.cm.jet(i / n))
    return colors


def hms(seconds, pos):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return '%d:%02d' % (h, m)


def plot_train_loss(title, train_data, batch_size, ax, x_ticks=True, y_ticks=True, plot_title=False,
                  plot_labels_bot=True, plot_labels_left=True, plot_labels_right=True):
    """
    NumIters,Seconds,LearningRate,loss
    """

    if x_ticks and y_ticks:
        ax.grid('on', axis='both', color='black', linewidth=0.1)

    iters = map(lambda x: x*batch_size, train_data.index.values.tolist()[5:])
    time = train_data['Seconds'].tolist()[5:]
    lr = train_data['LearningRate'].tolist()[5:]

    # test_data_len = len(test_data['Accuracy'].tolist())
    # elems = int((len(iters) - 1) / test_data_len)
    colors = iter(jet_colors(3))

    train_loss_data = train_data['loss'].tolist()[5:]
    ax.plot(iters, train_loss_data, label='TrainingLoss', color=colors.next(),
             marker='.', linestyle='-', alpha=0.6, linewidth=0.3, markevery=1, markeredgewidth=0.2, markersize=3)
    # ax.plot(iters, lr, label='LearningRate', color=colors.next(),
    #         marker='x', linestyle='-', alpha=0.6, linewidth=0.3, markevery=1, markeredgewidth=0.2, markersize=3)
    if plot_labels_bot:
        ax.set_xlabel('Examples seen')
    if plot_labels_left:
        ax.set_ylabel('Train Loss')

    ax3 = ax.twiny()
    # ax3.set_xlabel('Time')
    ax3.xaxis.set_major_formatter(plot.FuncFormatter(hms))
    ax3.xaxis.set_ticks(time)

    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda k, p: '%dM' % (k / 1000000)))

    ax.set_xticks(np.linspace(0, ax.get_xbound()[1], 6))
    ax3.set_xticks(np.linspace(0, ax3.get_xbound()[1], 6))

    if plot_title:
        # ax.set_title('$\\textsc{' + title + '}$', y=1.1)
        ax.set_title(title, y=1.1)

    ax.xaxis.set_tick_params(length=2)
    ax.yaxis.set_tick_params(length=2)

    # for label in ax2.xaxis.get_ticklabels()[::2]:
    #     label.set_visible(False)

    [i.set_linewidth(0.5) for i in ax.spines.itervalues()]
    [i.set_linewidth(0.5) for i in ax3.spines.itervalues()]

    handles1, labels1 = ax.get_legend_handles_labels()
    # handles2, labels2 = ax2.get_legend_handles_labels()
    return handles1, labels1


def plot_test_loss_acc(title, test_data, batch_size, ax, x_ticks=True, y_ticks=True, plot_title=True,
                  plot_labels_bot=True, plot_labels_left=True, plot_labels_right=True):
    if x_ticks and y_ticks:
        ax.grid('on', axis='both', color='black', linewidth=0.1)

    iters = map(lambda x: x*batch_size, test_data.index.values.tolist()[5:])
    time = test_data['Seconds'].tolist()[5:]

    colors = iter(jet_colors(3))

    loss = test_data['loss'].tolist()[5:]
    ax.plot(iters, loss, label='TestLoss', color=colors.next(),
            marker='.', linestyle='-', alpha=0.6, linewidth=0.3, markevery=1, markeredgewidth=0.2, markersize=3)
    if plot_labels_bot:
        ax.set_xlabel('Examples seen')
    if plot_labels_left:
        ax.set_ylabel('Test Loss')

    ax2 = ax.twinx()
    ax2.plot(iters, test_data['acc'].tolist()[5:], label='TestAccuracy', color=colors.next(),
             marker='.', linestyle='-', alpha=0.6, linewidth=0.3, markevery=1, markeredgewidth=0.2, markersize=2)
    if plot_labels_right:
        ax2.set_ylabel('Test Accuracy')

    ax3 = ax.twiny()
    # ax3.set_xlabel('Time')
    ax3.xaxis.set_major_formatter(plot.FuncFormatter(hms))
    ax3.xaxis.set_ticks(time)

    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda k, p: '%dM' % (k / 1000000)))

    ax.set_xticks(np.linspace(0, ax.get_xbound()[1], 6))
    ax3.set_xticks(np.linspace(0, ax3.get_xbound()[1], 6))

    if plot_title:
        # ax.set_title('$\\textsc{' + title + '}$', y=1.1)
        ax.set_title(title, y=1.1)

    ax.xaxis.set_tick_params(length=2)
    ax.yaxis.set_tick_params(length=2)

    for label in ax2.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)

    [i.set_linewidth(0.5) for i in ax.spines.itervalues()]
    [i.set_linewidth(0.5) for i in ax2.spines.itervalues()]
    [i.set_linewidth(0.5) for i in ax3.spines.itervalues()]

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    return handles1 + handles2, labels1 + labels2


def plot_all(nets_dir, out, cols=4, share_x=True, share_y=False, figsize=(8, 9)):
    def net_names():
        return sorted([name for name in os.listdir(nets_dir) if os.path.isdir(os.path.join(nets_dir, name))])

    train_test_data = [parse_log(os.path.join(nets_dir, net, 'cnn_nopool/log.txt')) + (net, ) for net in net_names()]

    nets_in_cols = cols / 2
    num_rows = len(train_test_data) / nets_in_cols
    f, axes = plot.subplots(int(num_rows), int(cols), sharex=share_x, sharey=share_y, figsize=figsize)

    for i, axs in enumerate(axes[0::2]):
        for ax_even, data in izip_longest(axs, train_test_data[i*cols:i*cols+cols]):
            if data is None:
                ax_even.axis('off')
                continue
            train_d, test_d, batch_size, net_name = data
            handles1, labels1 = plot_train_loss(net_name, train_d, batch_size, ax_even)

    for i, axs in enumerate(axes[1::2]):
        for ax_odd, data in izip_longest(axs, train_test_data[i*cols:i*cols+cols]):
            if data is None:
                ax_odd.axis('off')
                continue
            train_d, test_d, batch_size, net_name = data
            handles2, labels2 = plot_test_loss_acc(net_name, test_d, batch_size, ax_odd, plot_title=False)


    # ax1 = axes[1][0]
    for i, axs in enumerate(axes[1::2]):
        ax1 = axs[0]
        for axx in axs[1:]:
            ax1.get_shared_y_axes().join(ax1, axx)
            ax1.get_shared_x_axes().join(ax1, axx)
    # legend
    # lgd = plot.figlegend(handles1 + handles2, labels1 + labels2, 'lower center', bbox_to_anchor=(0.45, 0.0), fancybox=True, shadow=False, ncol=6)
    # lgd.get_frame().set_linewidth(0.0)

    plot.subplots_adjust(bottom=0.14)
    plot.subplots_adjust(hspace=0.4)
    plot.subplots_adjust(wspace=0.8)

    # this make subplots_adjust unnecesary (no overlapping) but u cant adjust legend position!
    # f.tight_layout()

    d = os.path.dirname(out)
    if not os.path.exists(d):
        os.makedirs(d)

    plot.savefig(out)


def plot_acc(paths, names):
    """
    NumIters,Seconds,LearningRate,loss

    """
    frames = []
    for path, name in zip(paths, names):
        train_data, test_data, batch_size = parse_log(path)
        # make 'numiters' normal column
        test_data.reset_index(level=0, inplace=True)
        # turn it examples seen
        test_data['NumIters'] *= batch_size
        # do some renaming
        test_data.rename(columns={'NumIters': 'ExamplesSeen'}, inplace=True)
        # test_data.columns = map(lambda col_name: name+ '-'+col_name, test_data.columns)
        # concatenate
        test_data['alg'] = name
        test_data['unit'] = 1
        frames.append(test_data)

    df = pd.concat(frames, ignore_index=True)
    #sns.palplot(sns.color_palette("Set2", 10))
    ax = sns.tsplot(df, time='ExamplesSeen', value='acc', condition='alg', unit='unit')
    ax.set_ylim(0.46, 0.60)
    ax.set_xlim(0, 80000000)
    plot.savefig('out.pdf')

if __name__ == '__main__':
    from parse_log import parse_log

    paths = [
        '/home/pliskowski/Documents/experiments/othello/18-11-2015/no_pool_exp2e/cnn_nopool/log.txt',
        '/home/pliskowski/Documents/experiments/BEAM/othello/18-11-2015/no_pool_exp2c/cnn_nopool/log.txt',
        '/home/pliskowski/Documents/experiments/BEAM/othello/19-11-2015/no_pool_exp2f/cnn_nopool/log.txt',
        '/home/pliskowski/Documents/experiments/BEAM/othello/18-11-2015/no_pool_exp2b/cnn_nopool/log.txt',
    ]
    names = [
        'exp2e',
        'exp2c',
        'exp2f',
        'exp2b'
    ]

    plot_acc(paths, names)

    # train_dict_list, test_dict_list, batch_size = parse_log('/home/pliskowski/Documents/experiments/othello/18-11-2015/no_pool_exp2e/cnn_nopool/log.txt')
    # fig = plot.figure(1, figsize=(5, 3))
    # ax = fig.add_subplot(1, 2, 1)
    #
    # # plot_train_loss('ss', train_dict_list, 256, ax)
    # plot_test_loss_acc('ss', test_dict_list, 256, ax)
    # fig.savefig('test2.pdf', bbox_inches='tight', pad_inches=0, dpi=300)
    # plot.close()

    # plot_all('/home/pliskowski/Documents/experiments/BEAM/othello/18-11-2015/', './nets.pdf', share_x=True, share_y=True)