import os
import numpy as np

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

from util.metrics import kappa
from util.stats import ci, se
from util.utils import create_dir


class ReportBuilder(object):

    def __init__(self, exp_dir=None):
        self.exp_dir = exp_dir
        self.opt_threshs = {}
        self.base_table = []
        self.precision_table = []
        self.recall_table = []
        self.f_score_table = []
        self.support_table = []
        self.arch_table = []

    def add(self, model_name, y_true, y_pred, y_score, acc_thresh=None, time=False):
        """
        :param y_true: True binary labels in range {0, 1}
        :param y_pred: Predicted labels, as returned by a classifier
        :param y_score: probability estimates of the positive class
        :return:
        """
        acc = round(accuracy_score(y_true, y_pred), 4)
        opt_acc = round(accuracy_score(y_true, y_score > acc_thresh), 4) if acc_thresh else 0
        kapp = round(kappa(y_true, y_pred, weights='quadratic'), 4)

        self.base_table.append({
            'model': model_name,
            'acc': acc,
            'opt-acc': opt_acc if acc_thresh is not None else 0,
            'kappa': kapp,
        })

        precision, recall, f_score, support = precision_recall_fscore_support(y_true, y_pred)

        def extract_row(perclass_data, model):
            d = {'class-{}'.format(class_id): class_res for class_id, class_res in enumerate(perclass_data)}
            d['model'] = model
            return d

        self.precision_table.append(extract_row(precision, model_name))
        self.recall_table.append(extract_row(recall, model_name))
        self.f_score_table.append(extract_row(f_score, model_name))
        self.support_table.append(extract_row(support, model_name))

    def generate(self, out_dir, loo=True):
        base_table = pd.DataFrame.from_records(self.base_table, index='model')
        precision_table = np.round(pd.DataFrame.from_records(self.precision_table, index='model'), 4)
        recall_table = np.round(pd.DataFrame.from_records(self.recall_table, index='model'), 4)
        f_score_table = np.round(pd.DataFrame.from_records(self.f_score_table, index='model'), 4)
        support_table = np.round(pd.DataFrame.from_records(self.support_table, index='model'), 4)

        all_tables = [(base_table, 'summary'), (precision_table, 'precision'), (recall_table, 'recall'),
                      (f_score_table, 'f_score'), (support_table, 'support')]

        save_results(all_tables, out_dir)

        if loo:
            dta = {col: [func(base_table[col]) for func in (np.mean, ci, np.std, se)] for col in base_table}
            stats = pd.DataFrame(dta, index=('mean', 'ci', 'std', 'se'))
            save_results([(stats, 'stats')], out_dir)


def save_results(all, out_dir):
    print 'Saving results to: {}'.format(out_dir)
    # create dirs
    csv_dir = os.path.join(out_dir, 'csv')
    # tex_dir = os.path.join(out_dir, 'tex')
    txt_dir = os.path.join(out_dir, 'txt')

    create_dir(out_dir)
    create_dir(csv_dir)
    # create_dir(tex_dir)
    create_dir(txt_dir)

    # save in several formats
    for table, name in all:
        table.to_csv(os.path.join(csv_dir, name + '.csv'))
        # table.to_latex(os.path.join(tex_dir, name + '.tex'))
        table.to_csv(os.path.join(txt_dir, name + '.txt'), sep='\t')
