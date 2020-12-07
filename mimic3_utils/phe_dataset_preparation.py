import os
import torch
import numpy as np
import math
import time
from mimic3_utils.mimic_3_files.readers import PhenotypingReader
from mimic3_utils.mimic_3_files.preprocessing import Discretizer, Normalizer
import mimic3_utils.data_reading_utils as data_reading_utils
import sys
import pathlib

class CustomDataset(torch.utils.data.BatchSampler):

    def __init__(self, reader, discretizer, normalizer,
                 batch_size, steps, shuffle, n_workers, return_names=False):
        self.reader = reader
        self.discretizer = discretizer
        self.normalizer = normalizer
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.return_names = return_names
        self.n_workers = n_workers

        if steps is None:
            self.n_examples = reader.get_number_of_examples()
            self.steps = (self.n_examples + batch_size - 1) // batch_size
        else:
            self.n_examples = steps * batch_size
            self.steps = steps

        self.chunk_size = min(128, self.steps) * batch_size
        self.remaining = self.n_examples

    def __len__(self):
        # return self.steps
        return math.ceil(self.n_examples / self.chunk_size)

    def __iter__(self):
        B = self.batch_size
        while self.remaining > 0:
            current_size = min(self.chunk_size, self.remaining)
            self.remaining -= current_size

            beg = time.time()
            print("reading next trunk")
            ret = self.reader.read_next_chunk(current_size)
            Xs = ret["X"]
            ts = ret["t"]
            ys = ret["y"]
            names = ret["name"]
            print("time spent:", time.time() - beg)

            Xs = data_reading_utils.preprocess_chunk_parallel(Xs, ts, self.discretizer, self.normalizer,
                                                              self.n_workers)
            yield (Xs, ys, ts, names)

    def on_epoch_end(self):
        if self.shuffle:
            self.reader.random_shuffle()
        self.remaining = self.n_examples


# the dir of to data after the steps in https://github.com/YerevaNN/mimic3-benchmarks
# in my case, the dir ends with "mimic3benchmark/scripts/data/phenotyping/"
# data_root = 'D:/yaquan/mimic3-benchmarks-master/mimic3benchmark/scripts/data/phenotyping/'
data_root = ''
save_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), 'phenotyping')
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
if data_root == '':
    print('Please fill in the data directory.')
    sys.exit()
print("Waring: the step is time consuming.")
timestep = 1.0
discretizer = Discretizer(timestep=timestep,
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')
n_worker = 40 # number pf workers to process the files in parallel
train_reader = PhenotypingReader(dataset_dir=os.path.join(data_root , 'train'),
                                  listfile=os.path.join(data_root , 'train_listfile.csv'),
                                  n_worker = n_worker)
val_reader = PhenotypingReader(dataset_dir=os.path.join(data_root , 'train'),
                                listfile=os.path.join(data_root , 'val_listfile.csv'),
                                n_worker=n_worker)
test_reader = PhenotypingReader(dataset_dir=os.path.join(data_root , 'test'),
                                 listfile=os.path.join(data_root , 'test_listfile.csv'),
                                 n_worker=n_worker)

discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
# save the headers
header_file = os.path.join(save_dir, 'header_list')
np.savez(header_file, discretizer_header)
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]
normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
# the normalizer files are copied from https://github.com/YerevaNN/mimic3-benchmarks
normalizer_state = 'ph_ts{}.input_str_previous.start_time_zero.normalizer'.format(timestep)
normalizer_state = os.path.join(pathlib.Path(__file__).parent.absolute(),
                                'mimic_3_files', 'normalizer', normalizer_state)
normalizer.load_params(normalizer_state)

data_reader_dict = {"train": train_reader, "val": val_reader, "test": test_reader}
batch_size = 512 # large enough to save all files
for data_type in data_reader_dict.keys():
    chunk_sample_size_list = []
    this_data_reader = data_reader_dict[data_type]
    data_loader_wrapper = CustomDataset(this_data_reader, discretizer, normalizer, batch_size,
                                        steps=None, shuffle=True, n_workers=20)
    for index, data in enumerate(data_loader_wrapper):
        X = np.array(data[0])
        print(X.shape)
        Y = np.array(data[1])
        np.savez_compressed(os.path.join(save_dir, data_type), X, Y)

