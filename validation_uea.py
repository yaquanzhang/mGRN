import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either &quot;0&quot; or &quot;1&quot;
gpu_index = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
print(gpu_index)
import uea_utils.utils as utils
import uea_utils.param_dict as param_dict
import numpy as np
import torch
import get_model
import check_accuracy
import pathlib

def single_model_validation(model_dir, model_param_dict, X_test, y_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_test = torch.tensor(X_test.copy()).to(device)
    y_test = torch.tensor(y_test.copy()).to(device)
    model_param_dict['device'] = device
    batch_first = model_param_dict['batch_first']
    num_classes = y_test.max().cpu().numpy() + 1
    n_feature = X_test.shape[2]
    model_param_dict['num_classes'] = num_classes
    model_param_dict['n_feature'] = n_feature
    if model_param_dict['input_size_list'] == 'total_split':
        model_param_dict['input_size_list'] = [1] * n_feature
    if not batch_first:
        X_test = X_test.permute(1, 0, 2)
    y_test = y_test.long().squeeze()
    X_test = X_test.float()
    print('print(model_param_dict)', model_param_dict)
    model = get_model.get_model(**model_param_dict)
    model.load_state_dict(torch.load(model_dir))
    _, validation_acc = check_accuracy.check_accuracy_single_model_tapnet(
        model, X_test, y_test)

if __name__ == '__main__':
    ##################### hyper parameters
    dataset = 'CharacterTrajectories'
    # dataset = 'FaceDetection'
    # dataset = 'LSST'
    # dataset = 'PenDigits'
    # dataset = 'PhonemeSpectra'
    # dataset = 'SpokenArabicDigits'
    data_root_dir = os.path.join(pathlib.Path(__file__).parent.absolute(),'uea_utils', 'uea_datasets')
    model_param_dict = param_dict.model_param_dict[dataset]
    training_param_dict = param_dict.training_param_dict[dataset]
    pretrained_model_path = os.path.join(pathlib.Path(__file__).parent.absolute(),
                                         'uea_pretrained_models', dataset + '.pth')
    #####################
    print(dataset)
    root_dir = os.path.join(data_root_dir, dataset)
    data_dir = os.path.join(root_dir, 'train.npz')
    data = np.load(data_dir)
    X_train = np.nan_to_num(data['arr_0'], nan=0.0)
    Y_train = data['arr_1']
    data_dir = os.path.join(root_dir, 'test.npz')
    data = np.load(data_dir)
    X_test = np.nan_to_num(data['arr_0'], nan=0.0)
    Y_test = data['arr_1']
    X_train, X_test = utils.normalization(X_train, X_test)
    single_model_validation(pretrained_model_path, model_param_dict, X_test, Y_test)
