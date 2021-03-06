# Mimic 3 datasets. in hospital mortality
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either &quot;0&quot; or &quot;1&quot;
gpu_index = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
import mimic3_utils.common_utils as common_utils
import numpy as np
import torch
import math
import time
import get_model
import pandas as pd
import train_model
from sklearn import metrics
import random
import sklearn.utils as sk_utils
import pathlib


class CustomDataset(torch.utils.data.BatchSampler):
    # to laod pre-saved chunk data
    def __init__(self, data_root, batch_size,
                 input_size_list_raw, device, max_length, shuffle = True):
        chunk_file_names = os.listdir(data_root)
        self.chunk_file_names = [x for x in chunk_file_names if 'chunk_sample_sizes' not in x]

        self.data_root = data_root
        chunk_sample_list_list = np.load(os.path.join(self.data_root, 'chunk_sample_sizes.npz'), allow_pickle=True)
        chunk_sample_list_list = chunk_sample_list_list['arr_0']

        total_sample_size = sum(chunk_sample_list_list)
        self.input_size_list_raw = input_size_list_raw
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

        self.max_length = max_length
        self.n_examples = total_sample_size
        self.steps = math.ceil(self.n_examples / batch_size)

        self.current_chunk_file_index = 0
        self.on_epoch_end()

        ###
        inf = 1e18
        self.bins = [(-inf, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 14), (14, +inf)]
        self.nbins = len(self.bins)

    def __len__(self):
        return self.steps

    def __iter__(self):
        B = self.batch_size
        while self.current_chunk_file_index < len(self.chunk_file_names):
            this_chunk_file_name = self.chunk_file_names[self.current_chunk_file_index]
            self.current_chunk_file_index = self.current_chunk_file_index + 1

            data = np.load(os.path.join(self.data_root, this_chunk_file_name), allow_pickle=True)
            Xs = data["arr_0"].tolist()
            ys = data["arr_1"].tolist()
            data.close()

            (Xs, ys) = common_utils.sort_and_shuffle([Xs, ys], B)
            current_size = len(Xs)

            # pad zero
            dtype = Xs[0].dtype
            Xs_padded = np.zeros((len(Xs), self.max_length, Xs[0].shape[1]), dtype=dtype)
            for i, x in enumerate(Xs):
                this_length = min(x.shape[0], self.max_length)
                Xs_padded[i, :this_length, :] = x[-this_length:]

            Xs_swapped = np.zeros(Xs_padded.shape, Xs_padded[0].dtype)
            beg_index = 0
            for sub_list in self.input_size_list_raw:
                this_size = len(sub_list)
                Xs_swapped[:, :, beg_index:(beg_index + this_size)] = Xs_padded[:, :, sub_list]
                beg_index = beg_index + this_size
            Xs_swapped = Xs_swapped.astype('float32')

            for i in range(0, current_size, B):
                X = torch.tensor(Xs_swapped[i:i + B]).to(self.device).float()
                X = X.permute(1, 0, 2)
                # X = self.shift_inputs(X)
                y = np.array(ys[i:i + B])
                y_processed = np.array([self.get_bin_custom(x) for x in y])
                y_processed = torch.tensor(y_processed).to(self.device).long()
                batch_data = (X, y_processed, y)
                yield batch_data

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.chunk_file_names)
        self.current_chunk_file_index = 0

    def get_bin_custom(self, x):
        for i in range(self.nbins):
            a = self.bins[i][0] * 24.0
            b = self.bins[i][1] * 24.0
            if a <= x < b:
                return i
        return None


class CheckAccuracy:
    def __init__(self, criterion, device, is_print = True):
        self.criterion = criterion
        self.is_print = is_print
        self.device = device
        # copied from Harutyunyan et al. (2019)
        self.bin_means = [11.450379, 35.070846, 59.206531, 83.382723, 107.487817,
              131.579534, 155.643957, 179.660558, 254.306624, 585.325890]

    def get_metrics(self, Y_array, predictions, Y_array_orig):
        kappa = metrics.cohen_kappa_score(Y_array, predictions, weights='linear')
        # We follow Harutyunyan et al. (2019) in the calculation of mad
        predictions_means = np.array([self.bin_means[int(x)] for x in predictions])
        mad = metrics.mean_absolute_error(Y_array_orig, predictions_means)
        return kappa, mad

    def check_accuracy(self, model, test_data, n_resample = None):
        # test_data: data loader
        model.eval()
        Y_test_list = []
        output_list = []
        Y_test_orig_list = []
        with torch.no_grad():
            for i, this_batch in enumerate(test_data):
                minibatch_X = this_batch[0]
                minibatch_Y = this_batch[1]
                minibatch_Y_orig_cpu = this_batch[2]
                minibatch_Y_cpu = minibatch_Y.cpu().numpy()
                outputs = model(minibatch_X)
                Y_test_list.append(minibatch_Y_cpu)
                Y_test_orig_list.append(minibatch_Y_orig_cpu)
                output_list.append(outputs.cpu())
        test_data.on_epoch_end()
        validation_outputs = torch.cat(output_list, dim=0).to(self.device)
        Y_test = np.concatenate(Y_test_list, axis=0)
        Y_test = torch.tensor(Y_test).to(self.device).long()
        if validation_outputs.shape[1] == 1:
            validation_outputs = validation_outputs.view(validation_outputs.shape[0])
        validation_loss = self.criterion(validation_outputs, Y_test).item()
        Y_array_cpu = Y_test.cpu().numpy()
        predictions_probability = torch.nn.functional.softmax(validation_outputs, dim=1).cpu().numpy()
        predictions = predictions_probability.argmax(axis=1)
        overall_acc = ((predictions == Y_array_cpu).sum() / Y_array_cpu.shape[0]).item()

        # metrics
        Y_array_orig = np.concatenate(Y_test_orig_list, axis=0)
        kappa, mad = self.get_metrics(Y_array_cpu, predictions, Y_array_orig)

        result_dict = {'loss': validation_loss, 'accuracy': overall_acc, 'kappa': kappa, 'mad': mad}
        if n_resample is None:
            if self.is_print:
                print("validation loss: {:.4f}".format(validation_loss))
                print("kappa: {:.4f}".format(kappa))
                print("mad: {:.4f}".format(mad))
        else:
            # resample to calculate confidence intervals
            print("resampling results")
            resample_result_list = []
            data = np.zeros((Y_array_cpu.shape[0], 3))
            data[:, 0] = np.array(Y_array_cpu)
            data[:, 1] = np.array(predictions)
            data[:, 2] = np.array(Y_array_orig)
            for i in range(n_resample):
                resample_data = sk_utils.resample(data, n_samples=len(data))
                kappa, mad = self.get_metrics(resample_data[:, 0], resample_data[:, 1], resample_data[:, 2])
                resample_result_list.append({'kappa': kappa, 'mad': mad})
            resample_result = pd.DataFrame(resample_result_list)
            for metric in ['kappa', 'mad']:
                # update the point value by mean
                result_dict[metric] = resample_result[metric].mean()
                result_dict[metric + '_lower'] = resample_result[metric].quantile(0.025)
                result_dict[metric + '_upper'] = resample_result[metric].quantile(0.975)
            if self.is_print:
                print(result_dict)

        return result_dict

def single_model(result_dir_root, model_param_dict, train_data_dir, val_data_dir, training_param_dict,
                 input_size_list_raw, max_length, N_trial = 1):
    print(result_dir_root)
    ########################### model training
    print("training...")
    model_saving_dir = os.path.join(result_dir_root, 'model')
    if not os.path.exists(model_saving_dir):
        os.makedirs(model_saving_dir)
    # data preparation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = training_param_dict['batch_size']
    del training_param_dict['batch_size']
    train_data = CustomDataset(train_data_dir, batch_size, input_size_list_raw, device, max_length)
    val_data = CustomDataset(val_data_dir, batch_size, input_size_list_raw, device, max_length, shuffle = False)
    model_param_dict['device'] = device

    print('print(model_param_dict)', model_param_dict)
    print('print(training_param_dict)', training_param_dict)
    np.savez(os.path.join(model_saving_dir, 'param_dict'), model_param_dict, training_param_dict)

    model = get_model.get_model(**model_param_dict)
    criterion = torch.nn.CrossEntropyLoss()
    check_accuracy_obj = CheckAccuracy(criterion, device)
    print('The number of trainable parameters is', model.param_num)
    val_result = train_model.train_mimic3(model, train_data, val_data, model_saving_dir, criterion,
                                          check_accuracy_obj,
                                          **training_param_dict)
    print('Validation result', val_result)

    accuray_result = pd.DataFrame([val_result])
    accuray_result.to_excel(os.path.join(result_dir_root, "accuracy_validation.xlsx"))


if __name__ == '__main__':
    #### parameters
    task = 'length_of_stay'
    data_root_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), 'mimic3_utils', task)  # data folder
    result_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), 'mimic3', task)  # save results in this folder
    # The experiments in Harutyunyan et al. (2019) are coded with Keras.
    # We enable Karas initialization so that results are comparable.
    model_param_dict = {"model_name": 'mGRN', "n_feature": 76, "n_rnn_units": 32,
                        "num_classes": 10, "batch_first": False,
                        "size_of": 8, "dropouti": 0.3, "dropoutw": 0, "dropouto": 0.3,
                        "keras_initialization": True}
    training_param_dict = {'batch_size': 1024, 'learning_rate': 1e-4, 'weight_decay':1e-7,
                           'num_epochs': 50, 'lr_decay_loss': 1.795, 'lr_decay_factor': 5,
                           'save_metric': 'kappa', 'save_model_starting_epoch': 5, }
    max_length = 72  # max number of time steps used for classification
    ####
    train_data_dir = os.path.join(data_root_dir, 'train')
    val_data_dir = os.path.join(data_root_dir, 'val')
    header_dir = os.path.join(data_root_dir, 'header_list.npz')
    # get the names of the columns
    header_data = np.load(header_dir)
    header = header_data['arr_0']
    header_data.close()
    # grouping of features
    input_size_list_raw = common_utils.get_input_size_raw(header)
    input_size_list = [len(x) for x in input_size_list_raw]
    model_param_dict['input_size_list'] = input_size_list
    single_model(result_dir, model_param_dict, train_data_dir, val_data_dir, training_param_dict,
                 input_size_list_raw, max_length)



