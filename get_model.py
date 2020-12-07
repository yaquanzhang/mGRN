# get models
import torch.nn as nn
import torch
import mGRNLayerDropout


class GRU(nn.Module):
    def __init__(self, n_feature, rnn_hidden_size, num_layers, num_classes, batch_first, device):
        super(GRU, self).__init__()
        self.hidden_size = rnn_hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.device = device
        self.gru = nn.GRU(n_feature, rnn_hidden_size, num_layers, batch_first=batch_first, bidirectional=False)
        self.fc1 = nn.Linear(rnn_hidden_size, num_classes)
        self.input_size = n_feature

    def get_param_num(self):
        # number of trainable parameters excluding output dense layer.
        rnn_units = self.hidden_size
        n_feature = self.input_size
        n_para = rnn_units*rnn_units*3 + n_feature*rnn_units*3 + 3*rnn_units
        return n_para

    def forward(self, x):
        # Set initial hidden and cell states
        if self.batch_first:
            batch_size = x.size(0)
        else:
            batch_size = x.size(1)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        # Forward propagate
        out, _ = self.gru(x, h0)  # pytorch native gru
        if len(out.shape) == 3:
            if self.batch_first:
                out = self.fc1(out[:, -1, :])
            else:
                out = self.fc1(out[-1, :, :])
        else:
            out = self.fc1(out)
        # out = self.sigmoid(out)
        return out


class LSTM(nn.Module):
    def __init__(self, n_feature, rnn_hidden_size, num_layers, num_classes, batch_first, device):
        super(LSTM, self).__init__()
        self.hidden_size = rnn_hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.device = device
        self.input_size = n_feature
        self.lstm = nn.LSTM(n_feature, rnn_hidden_size, num_layers, batch_first=batch_first, bidirectional = False)
        self.fc1 = nn.Linear(rnn_hidden_size, num_classes)
        # self.sigmoid = nn.Sigmoid()

    def get_param_num(self):
        # number of trainable parameters excluding output dense layer.
        rnn_units = self.hidden_size
        n_feature = self.input_size
        n_para = rnn_units*rnn_units*4 + n_feature*rnn_units*4 + 4*rnn_units
        return n_para

    def forward(self, x):
        # Set initial hidden and cell states
        if self.batch_first:
            batch_size = x.size(0)
        else:
            batch_size = x.size(1)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        # forward propogate
        out, _ = self.lstm(x, (h0, c0)) # lstm
        if len(out.shape) == 3:
            if self.batch_first:
                out = self.fc1(out[:, -1, :])
            else:
                out = self.fc1(out[-1, :, :])
        else:
            out = self.fc1(out)
        # out = self.sigmoid(out)
        return out


class ChannelwiseLSTM(nn.Module):
    def __init__(self, rnn_hidden_size, num_layers, num_classes, batch_first, device,
                 input_size_list, size_of = 1):
        super(ChannelwiseLSTM, self).__init__()
        self.hidden_size = rnn_hidden_size
        self.batch_first = batch_first
        self.input_size_list = input_size_list
        self.num_layers = num_layers
        self.size_of = size_of
        self.forward_marginal_list = mGRNLayerDropout.ListModule(self, 'marginal_forward_')
        self.backward_marginal_list = mGRNLayerDropout.ListModule(self, 'marginal_backward_')
        self.device = device
        self.joint_layer_size = int(size_of*rnn_hidden_size)
        for input_size in self.input_size_list:
            self.forward_marginal_list.append(nn.LSTM(input_size, rnn_hidden_size, num_layers,
                                                      batch_first=batch_first, bidirectional = False))
            self.backward_marginal_list.append(nn.LSTM(input_size, rnn_hidden_size, num_layers,
                                                      batch_first=batch_first, bidirectional=False))

        self.n_group = len(input_size_list)
        self.input_size = sum(list(input_size_list)) # number of total input features
        self.joint = nn.LSTM(self.n_group*rnn_hidden_size*2, self.joint_layer_size, num_layers, batch_first=batch_first,
                             bidirectional = False)
        self.fc1 = torch.nn.Linear(self.joint_layer_size, num_classes)

    def get_param_num(self):
        # number of trainable parameters excluding output dense layer.
        rnn_units = self.hidden_size
        n_group = self.n_group
        n_feature = self.input_size
        size_of = self.size_of
        n_para = ((rnn_units*rnn_units*4 + 4*rnn_units)*n_group + n_feature*rnn_units*4)*2 + \
                    rnn_units*rnn_units*4*size_of*size_of + (2*n_group*rnn_units)*rnn_units*4*size_of + \
                 4*rnn_units*size_of
        return n_para

    def forward(self, x):
        if self.batch_first:
            batch_size = x.size(0)
            backward_x = torch.flip(x, dims = [1])
        else:
            # x : (T_dimension, batch_size, n_features)
            batch_size = x.size(1)
            backward_x = torch.flip(x, dims = [0])
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        input_index_beg = 0
        out_list = []
        for i, input_size in enumerate(self.input_size_list):
            forward_mrginal = self.forward_marginal_list[i]
            backward_marginal = self.backward_marginal_list[i]
            input_index_end = input_size + input_index_beg
            this_x = x[:, :, input_index_beg:input_index_end]
            this_backward_x = backward_x[:, :, input_index_beg:input_index_end]
            forward_out, _ = forward_mrginal(this_x, (h0, c0))  # lstm
            backward_out, _ = backward_marginal(this_backward_x, (h0, c0))
            backward_out = torch.flip(backward_out, dims=[0])
            input_index_beg = input_index_end
            out_list = out_list + [forward_out, backward_out]

        h0 = torch.zeros(self.num_layers, batch_size, self.joint_layer_size).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.joint_layer_size).to(self.device)
        out = torch.cat(out_list, dim=2)
        # out = self.dropout(out)
        out, _ = self.joint(out, (h0, c0))
        if len(out.shape) == 3:
            if self.batch_first:
                out = out[:, -1, :]
            else:
                out = out[-1, :, :]
        # out = self.dropout(out)
        out = self.fc1(out)
        return out


def get_model(model_name, n_feature, n_rnn_units, num_classes, batch_first, device,
              n_layers = 1, input_size_list = None,
              size_of = 1, dropouti = 0, dropoutw = 0, dropouto = 0,
              keras_initialization = False):
    '''
    :param model_name: the name of the model
    :return model object
    '''
    if model_name == "LSTM":
        model = LSTM(n_feature, n_rnn_units, n_layers, num_classes, batch_first, device).to(device)
    elif model_name == "GRU":
        model = GRU(n_feature, n_rnn_units, n_layers, num_classes, batch_first, device).to(device)
    elif "mGRN" in model_name:
        model = mGRNLayerDropout.mGRNDropout(n_rnn_units, num_classes, device, input_size_list,
                                             size_of, dropouti, dropoutw, dropouto,
                                             keras_initialization).to(device)
    elif model_name == "ChannelwiseLSTM":
        model = ChannelwiseLSTM(n_rnn_units, n_layers, num_classes, batch_first, device,
                              input_size_list, size_of).to(device)
    else:
        print("Error: unrecognized model name in get model")
        model = None
    return model




