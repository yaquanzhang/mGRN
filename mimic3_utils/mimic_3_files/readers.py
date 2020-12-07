from __future__ import absolute_import
from __future__ import print_function

import os
import numpy as np
import random
from joblib import Parallel, delayed


class Reader(object):
    def __init__(self, dataset_dir, listfile=None, n_worker = 40):
        self._dataset_dir = dataset_dir
        self._current_index = 0
        if listfile is None:
            listfile_path = os.path.join(dataset_dir, "listfile.csv")
        else:
            listfile_path = listfile
        with open(listfile_path, "r") as lfile:
            self._data = lfile.readlines()
        self._listfile_header = self._data[0]
        self._data = self._data[1:]
        self.n_worker = n_worker

    def get_number_of_examples(self):
        return len(self._data)

    def random_shuffle(self, seed=None):
        if seed is not None:
            random.seed(seed)
        random.shuffle(self._data)

    def read_example(self, index):
        raise NotImplementedError()

    def read_next(self):
        to_read_index = self._current_index
        self._current_index += 1
        if self._current_index == self.get_number_of_examples():
            self._current_index = 0
        return self.read_example(to_read_index)
    
    def get_parallel_inputs(self, index_list):
        if len(self._data[index_list[0]]) == 3:
            inputs_list = [(os.path.join(self._dataset_dir, self._data[index][0]), self._data[index][1], self._data[index][2]) for index in index_list]
        else:
            inputs_list = [
                (os.path.join(self._dataset_dir, self._data[index][0]), self._data[index][1]) for
                index in index_list]
        return inputs_list
    
    def read_next_chunk(self, chunk_size):
        index_end = min(self._current_index+chunk_size, self.get_number_of_examples())
        to_read_index = range(self._current_index, self._current_index+chunk_size)
        print('reading index from ', to_read_index[0], ' to ', to_read_index[-1])
        self._current_index = index_end
        if self._current_index == self.get_number_of_examples():
            self._current_index = 0
        # read a chunk by parallelization
        print('file reading begins')
        inputs_list = self.get_parallel_inputs(to_read_index)
        item_list = Parallel(n_jobs=self.n_worker)(delayed(self.read_time_series_parallel)(this_inputs) for this_inputs in inputs_list)
        data = {}
        for ret_index, ret_raw in enumerate(item_list):
            if len(inputs_list[ret_index]) == 3:
                ret = {"X": ret_raw[0],
                    "t": inputs_list[ret_index][1],
                    "y": inputs_list[ret_index][2],
                    "header": ret_raw[1],
                    "name": inputs_list[ret_index][0]}
            else:
                ret = {"X": ret_raw[0],
                       "t": self._period_length,
                       "y": inputs_list[ret_index][1],
                       "header": ret_raw[1],
                       "name": inputs_list[ret_index][0]}
            for k, v in ret.items():
                if k not in data:
                    data[k] = []
                data[k].append(v)
        data["header"] = data["header"][0]
        print('file reading finished')
        return data


def Decompensation_read_timeseries(inputs):
    ts_filename = inputs[0]
    time_bound = inputs[1]
    ret = []
    with open(ts_filename, "r") as tsfile:
        header = tsfile.readline().strip().split(',')
        assert header[0] == "Hours"
        for line in tsfile:
            mas = line.strip().split(',')
            t = float(mas[0])
            if t > time_bound + 1e-6:
                break
            ret.append(np.array(mas))
    return (np.stack(ret), header)


class DecompensationReader(Reader):
    def __init__(self, dataset_dir, listfile=None, n_worker = 40):
        """ Reader for decompensation prediction task.
        :param dataset_dir: Directory where timeseries files are stored.
        :param listfile:    Path to a listfile. If this parameter is left `None` then
                            `dataset_dir/listfile.csv` will be used.
        """
        Reader.__init__(self, dataset_dir, listfile, n_worker)
        self._data = [line.split(',') for line in self._data]
        self._data = [(x, float(t), int(y)) for (x, t, y) in self._data]
        self.read_time_series_parallel = Decompensation_read_timeseries

    def _read_timeseries(self, ts_filename, time_bound):
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                t = float(mas[0])
                if t > time_bound + 1e-6:
                    break
                ret.append(np.array(mas))
        return (np.stack(ret), header)
    


    def read_example(self, index):
        """ Read the example with given index.

        :param index: Index of the line of the listfile to read (counting starts from 0).
        :return: Directory with the following keys:
            X : np.array
                2D array containing all events. Each row corresponds to a moment.
                First column is the time and other columns correspond to different
                variables.
            t : float
                Length of the data in hours. Note, in general, it is not equal to the
                timestamp of last event.
            y : int (0 or 1)
                Mortality within next 24 hours.
            header : array of strings
                Names of the columns. The ordering of the columns is always the same.
            name: Name of the sample.
        """
        if index < 0 or index >= len(self._data):
            raise ValueError("Index must be from 0 (inclusive) to number of examples (exclusive).")

        name = self._data[index][0]
        t = self._data[index][1]
        y = self._data[index][2]
        (X, header) = self._read_timeseries(name, t)

        return {"X": X,
                "t": t,
                "y": y,
                "header": header,
                "name": name}


def InHospitalMortality_read_timeseries(inputs):
    ts_filename = inputs[0]
    ret = []
    with open(ts_filename, "r") as tsfile:
        header = tsfile.readline().strip().split(',')
        assert header[0] == "Hours"
        for line in tsfile:
            mas = line.strip().split(',')
            ret.append(np.array(mas))
    return (np.stack(ret), header)


class InHospitalMortalityReader(Reader):
    def __init__(self, dataset_dir, listfile=None, period_length=48.0, n_worker = 40):
        """ Reader for in-hospital moratality prediction task.

        :param dataset_dir:   Directory where timeseries files are stored.
        :param listfile:      Path to a listfile. If this parameter is left `None` then
                              `dataset_dir/listfile.csv` will be used.
        :param period_length: Length of the period (in hours) from which the prediction is done.
        """
        Reader.__init__(self, dataset_dir, listfile, n_worker)
        self._data = [line.split(',') for line in self._data]
        self._data = [(x, int(y)) for (x, y) in self._data]
        self._period_length = period_length
        self.read_time_series_parallel = InHospitalMortality_read_timeseries

    def _read_timeseries(self, ts_filename):
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                ret.append(np.array(mas))
        return (np.stack(ret), header)

    def read_example(self, index):
        """ Reads the example with given index.

        :param index: Index of the line of the listfile to read (counting starts from 0).
        :return: Dictionary with the following keys:
            X : np.array
                2D array containing all events. Each row corresponds to a moment.
                First column is the time and other columns correspond to different
                variables.
            t : float
                Length of the data in hours. Note, in general, it is not equal to the
                timestamp of last event.
            y : int (0 or 1)
                In-hospital mortality.
            header : array of strings
                Names of the columns. The ordering of the columns is always the same.
            name: Name of the sample.
        """
        if index < 0 or index >= len(self._data):
            raise ValueError("Index must be from 0 (inclusive) to number of lines (exclusive).")

        name = self._data[index][0]
        t = self._period_length
        y = self._data[index][1]
        (X, header) = self._read_timeseries(name)

        return {"X": X,
                "t": t,
                "y": y,
                "header": header,
                "name": name}

    

def LengthOfStay_read_timeseries(inputs):
    ts_filename = inputs[0]
    time_bound = inputs[1]
    ret = []
    with open(ts_filename, "r") as tsfile:
        header = tsfile.readline().strip().split(',')
        assert header[0] == "Hours"
        for line in tsfile:
            mas = line.strip().split(',')
            t = float(mas[0])
            if t > time_bound + 1e-6:
                break
            ret.append(np.array(mas))
    return (np.stack(ret), header)

    

class LengthOfStayReader(Reader):
    def __init__(self, dataset_dir, listfile=None, n_worker = 40):
        """ Reader for length of stay prediction task.

        :param dataset_dir: Directory where timeseries files are stored.
        :param listfile:    Path to a listfile. If this parameter is left `None` then
                            `dataset_dir/listfile.csv` will be used.
        """
        Reader.__init__(self, dataset_dir, listfile, n_worker)
        self._data = [line.split(',') for line in self._data]
        self._data = [(x, float(t), float(y)) for (x, t, y) in self._data]
        self.read_time_series_parallel = LengthOfStay_read_timeseries

    def _read_timeseries(self, ts_filename, time_bound):
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                t = float(mas[0])
                if t > time_bound + 1e-6:
                    break
                ret.append(np.array(mas))
        return (np.stack(ret), header)

    def read_example(self, index):
        """ Reads the example with given index.

        :param index: Index of the line of the listfile to read (counting starts from 0).
        :return: Dictionary with the following keys:
            X : np.array
                2D array containing all events. Each row corresponds to a moment.
                First column is the time and other columns correspond to different
                variables.
            t : float
                Length of the data in hours. Note, in general, it is not equal to the
                timestamp of last event.
            y : float
                Remaining time in ICU.
            header : array of strings
                Names of the columns. The ordering of the columns is always the same.
            name: Name of the sample.
        """
        if index < 0 or index >= len(self._data):
            raise ValueError("Index must be from 0 (inclusive) to number of lines (exclusive).")

        name = self._data[index][0]
        t = self._data[index][1]
        y = self._data[index][2]
        (X, header) = self._read_timeseries(name, t)

        return {"X": X,
                "t": t,
                "y": y,
                "header": header,
                "name": name}


def Phenotyping_read_timeseries(inputs):
    ts_filename = inputs[0]
    ret = []
    with open(ts_filename, "r") as tsfile:
        header = tsfile.readline().strip().split(',')
        assert header[0] == "Hours"
        for line in tsfile:
            mas = line.strip().split(',')
            ret.append(np.array(mas))
    return (np.stack(ret), header)


class PhenotypingReader(Reader):
    def __init__(self, dataset_dir, listfile=None, n_worker = 40):
        """ Reader for phenotype classification task.

        :param dataset_dir: Directory where timeseries files are stored.
        :param listfile:    Path to a listfile. If this parameter is left `None` then
                            `dataset_dir/listfile.csv` will be used.
        """
        Reader.__init__(self, dataset_dir, listfile, n_worker)
        self._data = [line.split(',') for line in self._data]
        self._data = [(mas[0], float(mas[1]), list(map(int, mas[2:]))) for mas in self._data]
        self.read_time_series_parallel = Phenotyping_read_timeseries

    def _read_timeseries(self, ts_filename):
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                ret.append(np.array(mas))
        return (np.stack(ret), header)

    def read_example(self, index):
        """ Reads the example with given index.

        :param index: Index of the line of the listfile to read (counting starts from 0).
        :return: Dictionary with the following keys:
            X : np.array
                2D array containing all events. Each row corresponds to a moment.
                First column is the time and other columns correspond to different
                variables.
            t : float
                Length of the data in hours. Note, in general, it is not equal to the
                timestamp of last event.
            y : array of ints
                Phenotype labels.
            header : array of strings
                Names of the columns. The ordering of the columns is always the same.
            name: Name of the sample.
        """
        if index < 0 or index >= len(self._data):
            raise ValueError("Index must be from 0 (inclusive) to number of lines (exclusive).")

        name = self._data[index][0]
        t = self._data[index][1]
        y = self._data[index][2]
        (X, header) = self._read_timeseries(name)

        return {"X": X,
                "t": t,
                "y": y,
                "header": header,
                "name": name}


