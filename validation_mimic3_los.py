import os
gpu_index = 2
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
print(gpu_index)
import numpy as np
import NN_main_mimic3_los as NN_main
import mimic3_utils.common_utils as common_utils
import torch
import get_model
import pathlib

def single_model_validation(model_dir, model_param_dict, test_data_dir, input_size_list_raw, max_length,
                            batch_size, n_resample):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_data = NN_main.CustomDataset(test_data_dir, batch_size, input_size_list_raw, device,
                                 max_length, shuffle=False)
    model_param_dict['device'] = device
    print('print(model_param_dict)', model_param_dict)
    model = get_model.get_model(**model_param_dict)
    model.load_state_dict(torch.load(model_dir))
    criterion = torch.nn.CrossEntropyLoss()
    check_arruracy_obj = NN_main.CheckAccuracy(criterion, device)
    check_arruracy_obj.check_accuracy(model, test_data, n_resample = n_resample)

if __name__ == '__main__':
    ##################### hyper parameters
    task = 'length_of_stay'
    data_root_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), 'mimic3_utils', task)  # data folder
    max_length = 72  # max number of time steps used for classification
    # The experiments in Harutyunyan et al. (2019) are coded with Keras.
    # We enable Karas initialization so that results are comparable.
    model_param_dict = {"model_name": 'mGRN', "n_feature": 76, "n_rnn_units": 32,
                        "num_classes": 10, "batch_first": False,
                        "size_of": 8, "dropouti": 0.3, "dropoutw": 0, "dropouto": 0.3,
                        "keras_initialization": True}
    batch_size = 1024 # gradually load data, must be divided by 1024
    n_resample = 1000 # Q
    pretrained_model_path = os.path.join(pathlib.Path(__file__).parent.absolute(),
                                         'mimic3_pretrained_models', task + '.pth')
    #####################
    np.random.seed(0)
    test_data_dir = os.path.join(data_root_dir, 'test')
    header_dir = os.path.join(data_root_dir, 'header_list.npz')
    header_data = np.load(header_dir)
    header = header_data['arr_0']
    header_data.close()
    # grouping of features
    input_size_list_raw = common_utils.get_input_size_raw(header)
    input_size_list = [len(x) for x in input_size_list_raw]
    model_param_dict['input_size_list'] = input_size_list
    single_model_validation(pretrained_model_path, model_param_dict, test_data_dir,
                            input_size_list_raw, max_length, batch_size, n_resample)