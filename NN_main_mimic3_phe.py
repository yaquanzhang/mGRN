# Mimic 3 datasets. in hospital mortality
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either &quot;0&quot; or &quot;1&quot;
gpu_index = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
import mimic3_utils.common_utils as common_utils
import numpy as np
import torch
import math
import shutil
import time
import get_model
import pandas as pd
import train_model
from sklearn import metrics
import sklearn.utils as sk_utils
import pathlib

class CustomDataset(torch.utils.data.BatchSampler):
    def __init__(self, data_dir, batch_size, input_size_list_raw, device, max_length, batch_first = False, shuffle = True):
        data = np.load(data_dir, allow_pickle=True)
        X_array_raw = data['arr_0']
        Y_array = data['arr_1']
        dtype = X_array_raw[0].dtype
        Xs_padded = np.zeros((len(X_array_raw), max_length, X_array_raw[0].shape[1]), dtype=dtype)
        # since some time series are too long, we truncate them and keep only the last max_length steps.
        for i, x in enumerate(X_array_raw):
            this_length = min(x.shape[0], max_length)
            Xs_padded[i, :this_length, :] = x[-this_length:]

        X_array_raw = Xs_padded
        X_array = np.zeros(X_array_raw.shape) # shape: (Sample size, T, n_features)
        # change the sequence of the features
        beg_index = 0
        for sub_list in input_size_list_raw:
            this_size = len(sub_list)
            X_array[:, :, beg_index:(beg_index + this_size)] = X_array_raw[:, :, sub_list]
            beg_index = beg_index + this_size
        X_array = X_array.astype('float32')
        self.n_examples = X_array.shape[0]
        self.steps = math.ceil(self.n_examples/batch_size)
        self.X_array = torch.tensor(X_array.copy()).to(device).float()
        self.Y_array = torch.tensor(Y_array.copy()).to(device).long()
        if not batch_first:
            self.X_array = self.X_array.permute(1, 0, 2)
        self.shuffle = shuffle
        self.on_epoch_end()
        self.batch_size = batch_size

    def __iter__(self):
        # return permuted tensors.
        for i in range(0, self.n_examples, self.batch_size):
            X = self.X_array[:, i:i + self.batch_size]
            Y = self.Y_array[i:i + self.batch_size]
            yield (X, Y)

    def __len__(self):
        return self.steps

    def on_epoch_end(self):
        if self.shuffle:
            shuffle_index = torch.randperm(self.X_array.size(1))
            self.X_array = self.X_array[:, shuffle_index, :]
            self.Y_array = self.Y_array[shuffle_index]
        # self.index = 0

class CheckAccuracy:
    def __init__(self, criterion, device, is_print = True):
        self.criterion = criterion
        self.is_print = is_print
        self.device = device

    def get_metrics(self, Y_array, predictions_probability):
        auc_macro = metrics.roc_auc_score(Y_array, predictions_probability, average="macro")
        auc_micro = metrics.roc_auc_score(Y_array, predictions_probability, average="micro")
        return auc_macro, auc_micro


    def check_accuracy(self, model, test_data, n_resample = None):
        # test_data: data loadersa
        model.eval()
        Y_test_list = []
        output_list = []
        with torch.no_grad():
            for i, this_batch in enumerate(test_data):
                minibatch_X = this_batch[0]
                minibatch_Y = this_batch[1]
                minibatch_Y_cpu = minibatch_Y.cpu().numpy()
                outputs = model(minibatch_X)
                Y_test_list.append(minibatch_Y_cpu)
                output_list.append(outputs.cpu())
        test_data.on_epoch_end()
        validation_outputs = torch.cat(output_list, dim=0).to(self.device)
        Y_test = np.concatenate(Y_test_list, axis=0)
        Y_test = torch.tensor(Y_test).to(self.device).long()
        if validation_outputs.shape[1] == 1:
            validation_outputs = validation_outputs.view(validation_outputs.shape[0])
        validation_loss = self.criterion(validation_outputs, Y_test).item()
        
        Y_array_cpu = Y_test.cpu().numpy()
        predictions_probability = torch.sigmoid(validation_outputs).cpu().numpy()

        auc_macro, auc_micro = self.get_metrics(Y_array_cpu, predictions_probability)

        result_dict = {'loss': validation_loss,  'auc_macro': auc_macro, 'auc_micro': auc_micro}
        if n_resample is None:
            if self.is_print:
                print("validation loss: {:.4f}".format(validation_loss))
                print("validation auc_macro: {:.4f}".format(auc_macro))
                print("validation auc_micro: {:.4f}".format(auc_micro))
        else:
            # resample to calculate confidence intervals
            print("resampling results")
            resample_result_list = []
            data = np.zeros((Y_array_cpu.shape[0], Y_array_cpu.shape[1] + predictions_probability.shape[1]))
            data[:, 0:Y_array_cpu.shape[1]] = Y_array_cpu
            data[:, Y_array_cpu.shape[1]:] = predictions_probability
            for i in range(n_resample):
                resample_data = sk_utils.resample(data, n_samples=len(data))
                auc_macro, auc_micro = self.get_metrics(resample_data[:, 0:Y_array_cpu.shape[1]],
                                              resample_data[:, Y_array_cpu.shape[1]:])
                resample_result_list.append({'auc_macro': auc_macro, 'auc_micro': auc_micro})
            resample_result = pd.DataFrame(resample_result_list)
            for metric in ['auc_macro', 'auc_micro']:
                # update the point value by mean
                result_dict[metric] = resample_result[metric].mean()
                result_dict[metric + '_lower'] = resample_result[metric].quantile(0.025)
                result_dict[metric + '_upper'] = resample_result[metric].quantile(0.975)
            if self.is_print:
                print(result_dict)
        return result_dict


class KerasBinaryCrossentropy(torch.nn.Module):
    # a combination of 25 clas
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.BCELoss()

    def forward(self, preds, target):
        loss = 0
        predicts_probability = torch.sigmoid(preds)
        for i in range(self.num_classes):
            this_target = target[:, i].float()
            this_predicts = predicts_probability[:, i]
            loss = loss + self.cross_entropy(this_predicts, this_target)
        loss = loss / self.num_classes
        return loss


def single_model(result_dir_root, model_param_dict, train_data_dir, val_data_dir, max_length, training_param_dict,
                 input_size_list_raw):
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
    val_data = CustomDataset(val_data_dir, 256, input_size_list_raw, device, max_length, shuffle = False)
    model_param_dict['device'] = device

    print('print(model_param_dict)', model_param_dict)
    print('print(training_param_dict)', training_param_dict)
    np.savez(os.path.join(model_saving_dir, 'param_dict'), model_param_dict, training_param_dict)

    model = get_model.get_model(**model_param_dict)
    criterion = KerasBinaryCrossentropy(model_param_dict['num_classes'])
    check_accuracy_obj = CheckAccuracy(criterion, device)
    print('The number of trainable parameters is', model.param_num)
    val_result = train_model.train_mimic3(model, train_data, val_data, model_saving_dir, criterion,
                                          check_accuracy_obj, **training_param_dict)
    print('Validation result', val_result)

    accuray_result = pd.DataFrame([val_result])
    accuray_result.to_excel(result_dir_root + "accuracy_validation.xlsx")


if __name__ == '__main__':
    #### parameters
    task = 'phenotyping'
    data_root_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), 'mimic3_utils', task)  # data folder
    result_dir = os.path.join(pathlib.Path(__file__).parent.absolute(),'mimic3', task) # save results in this folder
    # The experiments in Harutyunyan et al. (2019) are coded with Keras.
    # We enable Karas initialization so that results are comparable.
    model_param_dict = {"model_name": 'mGRN', "n_feature": 76, "n_rnn_units": 32,
                        "num_classes": 25, "batch_first": False,
                        "size_of": 2, "dropouti": 0.1, "dropoutw": 0, "dropouto": 0.1,
                        "keras_initialization": True}
    training_param_dict = {'batch_size': 16, 'learning_rate': 1e-3, 'weight_decay':1e-5,
                            'num_epochs': 100, 'lr_decay_loss': None, 'lr_decay_factor': None,
                           'save_metric': 'loss', 'save_model_starting_epoch':10}
    max_length = 200
    ####
    train_data_dir = os.path.join(data_root_dir, 'train.npz')
    val_data_dir = os.path.join(data_root_dir, 'val.npz')
    header_dir = os.path.join(data_root_dir, 'header_list.npz')
    # get the names of the columns
    header_data = np.load(header_dir)
    header = header_data['arr_0']
    # grouping of features
    input_size_list_raw = common_utils.get_input_size_raw(header)
    input_size_list = [len(x) for x in input_size_list_raw]
    model_param_dict['input_size_list'] = input_size_list
    single_model(result_dir, model_param_dict, train_data_dir, val_data_dir, max_length, training_param_dict,
                 input_size_list_raw)



