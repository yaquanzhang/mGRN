from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import os
import json
import random

def read_chunk(reader, chunk_size):
    data = {}
    for i in range(chunk_size):
        ret = reader.read_next()
        for k, v in ret.items():
            if k not in data:
                data[k] = []
            data[k].append(v)
    data["header"] = data["header"][0]
    return data

def sort_and_shuffle(data, batch_size):
    """ Sort data by the length and then make batches and shuffle them.
        data is tuple (X1, X2, ..., Xn) all of them have the same length.
        Usually data = (X, y).
    """
    assert len(data) >= 2
    data = list(zip(*data))

    random.shuffle(data)

    old_size = len(data)
    rem = old_size % batch_size
    head = data[:old_size - rem]
    tail = data[old_size - rem:]
    data = []

    head.sort(key=(lambda x: x[0].shape[0]))

    mas = [head[i: i+batch_size] for i in range(0, len(head), batch_size)]
    random.shuffle(mas)

    for x in mas:
        data += x
    data += tail

    data = list(zip(*data))
    return data

def pad_zeros(arr, min_length=None):
    """
    `arr` is an array of `np.array`s

    The function appends zeros to every `np.array` in `arr`
    to equalize their first axis lenghts.
    """
    dtype = arr[0].dtype
    max_len = max([x.shape[0] for x in arr])
    ret = [np.concatenate([x, np.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
           for x in arr]
    if (min_length is not None) and ret[0].shape[0] < min_length:
        ret = [np.concatenate([x, np.zeros((min_length - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
               for x in ret]
    return np.array(ret)

def get_input_size_raw(header):
    # channel_wise_lstm line 38-54
    channel_names = set()
    for ch in header:
        if ch.find("mask->") != -1:
            continue
        pos = ch.find("->")
        if pos != -1:
            channel_names.add(ch[:pos])
        else:
            channel_names.add(ch)
    channel_names = sorted(list(channel_names))
    print("==> found {} channels: {}".format(len(channel_names), channel_names))
    channels = []  # each channel is a list of columns
    for ch in channel_names:
        indices = range(len(header))
        indices = list(filter(lambda i: header[i].find(ch) != -1, indices))
        channels.append(indices)
    return channels