#! /usr/bin/env python
# -*- coding: utf-8
'''
Python implementation of Krippendorff's alpha -- inter-rater reliability
(c)2011-17 Thomas Grill (http://grrrr.org)
Python version >= 2.4 required
'''




from __future__ import print_function
import pprint
import pandas as pd
from scipy import stats
import csv
import matplotlib

try:
    import numpy as np
except ImportError:
    np = None


def nominal_metric(a, b):
    return a != b


def interval_metric(a, b):
    return (a - b) ** 2


def ratio_metric(a, b):
    return ((a - b) / (a + b)) ** 2


def krippendorff_alpha(data, metric=interval_metric, force_vecmath=False, convert_items=float, missing_items=None):
    '''
    Calculate Krippendorff's alpha (inter-rater reliability):

    data is in the format
    [
        {unit1:value, unit2:value, ...},  # coder 1
        {unit1:value, unit3:value, ...},   # coder 2
        ...                            # more coders
    ]
    or
    it is a sequence of (masked) sequences (list, numpy.array, numpy.ma.array, e.g.) with rows corresponding to coders and columns to items

    metric: function calculating the pairwise distance
    force_vecmath: force vector math for custom metrics (numpy required)
    convert_items: function for the type conversion of items (default: float)
    missing_items: indicator for missing items (default: None)
    '''

    # number of coders
    m = len(data)

    # set of constants identifying missing values
    if missing_items is None:
        maskitems = []
    else:
        maskitems = list(missing_items)
    if np is not None:
        maskitems.append(np.ma.masked_singleton)

    # convert input data to a dict of items
    units = {}
    for d in data:
        try:
            # try if d behaves as a dict
            diter = d.items()
        except AttributeError:
            # sequence assumed for d
            diter = enumerate(d)

        for it, g in diter:
            if g not in maskitems:
                try:
                    its = units[it]
                except KeyError:
                    its = []
                    units[it] = its
                its.append(convert_items(g))

    units = dict((it, d) for it, d in units.items() if len(d) > 1)  # units with pairable values
    n = sum(len(pv) for pv in units.values())  # number of pairable values

    if n == 0:
        raise ValueError("No items to compare.")

    np_metric = (np is not None) and ((metric in (interval_metric, nominal_metric, ratio_metric)) or force_vecmath)

    Do = 0.
    for grades in units.values():
        if np_metric:
            gr = np.asarray(grades)
            Du = sum(np.sum(metric(gr, gri)) for gri in gr)
        else:
            Du = sum(metric(gi, gj) for gi in grades for gj in grades)
        Do += Du / float(len(grades) - 1)
    Do /= float(n)

    if Do == 0:
        return 1.

    De = 0.
    for g1 in units.values():
        if np_metric:
            d1 = np.asarray(g1)
            for g2 in units.values():
                De += sum(np.sum(metric(d1, gj)) for gj in g2)
        else:
            for g2 in units.values():
                De += sum(metric(gi, gj) for gi in g1 for gj in g2)
    De /= float(n * (n - 1))

    return 1. - Do / De if (Do and De) else 1.


if __name__ == '__main__':
    print("Example from http://en.wikipedia.org/wiki/Krippendorff's_Alpha")
    filename = 'f1255158.csv'

    df = pd.read_csv(filename)
    table = df.pivot_table( values='how_civil_was_the_reply_by_personb_', index=['_worker_id'],columns = ['post_id'])

    #  for calculate pearson
    tabl_nan= table.copy()
    array_nan = np.asarray(tabl_nan)

    table = table.replace(np.nan, '*', regex=True)
    array = []
    for row in table.iterrows():
        index, data = row
        array.append(data.tolist())
    array= np.asarray(array)

    # print(array)
    # print(array_nan)


    missing = '*'  # indicator for missing values
    # array = [d.split() for d in data]  # convert to 2D list of string items

    # print("nominal metric: %.3f" % krippendorff_alpha(array, nominal_metric, missing_items=missing))
    print("interval metric for online worker: %.3f" % krippendorff_alpha(array, interval_metric, missing_items=missing))

    # calculate pearson score:
    table_p = df.pivot_table(values='how_civil_was_the_reply_by_personb_', index=['post_id'], columns=['_worker_id'])
    correlation = table_p.corr(method='pearson', min_periods=1).round(3)
    step1_mean=correlation.mean()
    workers_mean= step1_mean.mean()
    print('workers mean correlation: ',round(workers_mean,3))
    with open('hw6_correlation_result.csv', 'w') as hw6_correlation_result: # , open('untrack_question.csv', 'w') as untrack_question
        writer = csv.writer(hw6_correlation_result)
        for each in correlation:
            row =list(correlation[each])
            writer.writerow(row)


    def get_array_group(filename):
        handle = open(filename, "r", encoding="utf8")
        lines = handle.read().split("\n")
        lst_u0 = []
        lst_u1 = []
        lst_u2 = []
        for line in lines[1:(len(lines) - 1)]:
            [u0, u1, u2] = line.split("\t")
            # print([u0, u1, u2])
            lst_u0.append(int(u0))
            lst_u1.append(int(u1))
            lst_u2.append(int(u2))
        return [lst_u0, lst_u1, lst_u2]


    group_data = get_array_group('group.tsv')


    correlation=[]
    for each1 in group_data:
        for each2 in group_data:
            if each1 != each2:
                x = np.array(each1)
                y = np.array(each2)


                corr= stats.pearsonr(np.array(each1), np.array(each2))
                correlation.append(corr[0])
    mean = np.mean(correlation)
    print("interval metric for group member: %.3f" % krippendorff_alpha(group_data, interval_metric))
    print('group mean correlation: ',round(mean,3))


