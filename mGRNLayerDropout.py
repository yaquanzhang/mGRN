from torch.nn.utils.rnn import PackedSequence
from torch.nn import Parameter
import torch.jit as jit
from typing import List, Tuple
from torch import Tensor
import torch
import torch.nn as nn
import math


class GRUDropoutCell(jit.ScriptModule):
    __constants__ = ['ngate']

    def __init__(self, input_size, hidden_size, dropoutw, keras_initialization):
        '''

        :param input_size: number of features in this marginal component
        :param hidden_size: number of hidden units of this marginal component
        :param dropoutw: recurrent dropout
        :param keras_initialization:
        '''
        super(GRUDropoutCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropoutw = dropoutw
        self.ngate = 3
        self.keras_initialization = keras_initialization
        self.w_ih = Parameter(torch.zeros(hidden_size * self.ngate, input_size))
        self.w_hh = Parameter(torch.zeros(hidden_size * self.ngate, hidden_size))
        self.b_ih = Parameter(torch.zeros(hidden_size * self.ngate))
        # self.b_hh = Parameter(torch.zeros(self.ngate * hidden_size))
        self.reset_parameters()

    def _drop_weights(self):
        # recurrent dropout
        for name, param in self.named_parameters():
            if 'w_hh' in name:
                getattr(self, name).data = \
                    torch.nn.functional.dropout(param.data, p=self.dropoutw,
                                                training=self.training).contiguous()

    @jit.script_method
    def forward(self, inputs, hidden):
        # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor]
        gi = torch.mm(inputs, self.w_ih.t()) + self.b_ih
        gh = torch.mm(hidden, self.w_hh.t())
        i_r, i_i, i_n = gi.chunk(self.ngate, 1)
        h_r, h_i, h_n = gh.chunk(self.ngate, 1)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy, newgate

    def reset_parameters(self):
        if self.keras_initialization:
            # Keras default initialization
            for name, param in self.named_parameters():
                if 'b' in name:
                    nn.init.constant_(param.data, 0)
                elif 'w_ih' in name:
                    nn.init.xavier_uniform_(param.data, gain=nn.init.calculate_gain('sigmoid'))
                elif 'w_hh' in name:
                    torch.nn.init.orthogonal_(param.data, gain=nn.init.calculate_gain('sigmoid'))
        else:
            # pytorch default initialization
            stdv = 1.0 / math.sqrt(self.hidden_size)
            for weight in self.parameters():
                nn.init.uniform_(weight, -stdv, stdv)


class GRUDropoutLayer(jit.ScriptModule):
    def __init__(self, *cell_args, batch_first, dropouti):
        '''
        GRU. newgates/temporary memory are exposed to be used in the joint layer.
        :param cell_args: args of the cell
        :param batch_first:
        :param dropouti: input dropout
        '''
        super(GRUDropoutLayer, self).__init__()
        self.cell = GRUDropoutCell(*cell_args)
        # self.dropoutw = dropoutw
        self.batch_first = batch_first
        #         self.input_drop = VariationalDropout(dropouti,
        #                                              batch_first=batch_first)
        self.dropout = dropouti

    def _input_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """
        https://github.com/keitakurita/Better_LSTM_PyTorch/blob/master/better_lstm/model.py
        Applies the same dropout mask across the temporal dimension
        See https://arxiv.org/abs/1512.05287 for more details.
        Note that this is not applied to the recurrent activations in the LSTM like the above paper.
        Instead, it is applied to the inputs and outputs of the recurrent layer.
        """
        if not self.training or self.dropout <= 0.:
            return x

        is_packed = isinstance(x, PackedSequence)
        if is_packed:
            x, batch_sizes = x
            if self.batch_first:
                max_batch_size = int(batch_sizes[0])
            else:
                max_batch_size = int(batch_sizes[1])
        else:
            batch_sizes = None
            if self.batch_first:
                max_batch_size = x.size(0)
            else:
                max_batch_size = x.size(1)

        # Drop same mask across entire sequence
        if self.batch_first:
            m = x.new_empty(max_batch_size, 1, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        else:
            m = x.new_empty(1, max_batch_size, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        x = x.masked_fill(m == 0, 0) / (1 - self.dropout)

        if is_packed:
            return PackedSequence(x, batch_sizes)
        else:
            return x

    # @jit.script_method
    def forward(self, inputs, out):
        # type: (Tensor, Tensor) -> Tensor
        if self.dropout>0:
            inputs = self._input_dropout(inputs)
        inputs = inputs.unbind(0)
        if self.cell.dropoutw > 0:
            self.cell._drop_weights()
        newgate_list = []
        for i in range(len(inputs)):
            out, newgate = self.cell(inputs[i], out)
            newgate_list.append(newgate)
        newgate_list = torch.stack(newgate_list)
        return out, newgate_list


class JointDropoutCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size, n_group, dropoutw, joint_hidden_size,
                 keras_initialization):
        '''
        :param input_size: number of total input features
        :param hidden_size: unit dimension of marginal components
        :param n_group: number of marginal components
        :param dropoutw: recurrent dropout
        :param joint_hidden_size: unit dimension of the joint component
        :param keras_initialization:
        '''
        super(JointDropoutCell, self).__init__()
        self.hidden_size = hidden_size
        self.keras_initialization = keras_initialization
        self.u_c = Parameter(torch.zeros(joint_hidden_size, hidden_size * n_group))
        self.b_c = Parameter(torch.zeros(joint_hidden_size))

        self.w_z = Parameter(torch.zeros(joint_hidden_size, input_size))
        self.u_z = Parameter(torch.zeros(joint_hidden_size, joint_hidden_size))
        self.b_z = Parameter(torch.zeros(joint_hidden_size))

        self.dropoutw = dropoutw

        self.reset_parameters()

    def _drop_weights(self):
        # recurrent dropout
        for name, param in self.named_parameters():
            if 'u_z' in name:
                getattr(self, name).data = \
                    torch.nn.functional.dropout(param.data, p=self.dropoutw,
                                                training=self.training).contiguous()

    @jit.script_method
    def forward(self, inputs, newgate, h_x):
        # type: (Tensor, Tensor, Tensor) -> Tensor
        newgate_combined = torch.mm(newgate, self.u_c.t()) + self.b_c
        newgate_combined = torch.tanh(newgate_combined)
        i_z = torch.mm(inputs, self.w_z.t()) + self.b_z  # W_{z}x + b_z
        h_z = torch.mm(h_x, self.u_z.t())
        inputgate_combined = torch.sigmoid(i_z + h_z)
        h_y = newgate_combined + inputgate_combined * (h_x - newgate_combined)
        return h_y

    def reset_parameters(self):
        if self.keras_initialization:
            print("Keras initialization")
            # Keras default initialization
            for name, param in self.named_parameters():
                if 'b' in name:
                    nn.init.constant_(param.data, 0)
                elif 'w' in name:
                    nn.init.xavier_uniform_(param.data, gain=nn.init.calculate_gain('sigmoid'))
                elif 'u' in name:
                    torch.nn.init.orthogonal_(param.data, gain=nn.init.calculate_gain('sigmoid'))
        else:
            # pytorch default initialization
            stdv = 1.0 / math.sqrt(self.hidden_size)
            for weight in self.parameters():
                nn.init.uniform_(weight, -stdv, stdv)


class JointDropoutLayer(jit.ScriptModule):
    def __init__(self, *cell_args, batch_first, dropouti):
        '''

        :param cell_args:
        :param batch_first:
        :param dropouti:
        '''
        super(JointDropoutLayer, self).__init__()
        self.cell = JointDropoutCell(*cell_args)
        # self.dropoutw = dropoutw
        self.batch_first = batch_first
        self.dropout = dropouti

    def _input_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """
        https://github.com/keitakurita/Better_LSTM_PyTorch/blob/master/better_lstm/model.py
        Applies the same dropout mask across the temporal dimension
        See https://arxiv.org/abs/1512.05287 for more details.
        Note that this is not applied to the recurrent activations in the LSTM like the above paper.
        Instead, it is applied to the inputs and outputs of the recurrent layer.
        """
        if not self.training or self.dropout <= 0.:
            return x

        is_packed = isinstance(x, PackedSequence)
        if is_packed:
            x, batch_sizes = x
            if self.batch_first:
                max_batch_size = int(batch_sizes[0])
            else:
                max_batch_size = int(batch_sizes[1])
        else:
            batch_sizes = None
            if self.batch_first:
                max_batch_size = x.size(0)
            else:
                max_batch_size = x.size(1)

        # Drop same mask across entire sequence
        if self.batch_first:
            m = x.new_empty(max_batch_size, 1, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        else:
            m = x.new_empty(1, max_batch_size, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        x = x.masked_fill(m == 0, 0) / (1 - self.dropout)

        if is_packed:
            return PackedSequence(x, batch_sizes)
        else:
            return x

    def forward(self, inputs, newgates, out):
        if self.dropout>0:
            inputs = self._input_dropout(inputs)
            newgates = self._input_dropout(newgates)
        inputs = inputs.unbind(0)
        newgates = newgates.unbind(0)
        if self.cell.dropoutw > 0:
            self.cell._drop_weights()
        for i in range(len(inputs)):
            # print(newgates[i].shape)
            out = self.cell(inputs[i], newgates[i], out)
        return out


class ListModule(object):
    # Should work with all kind of module
    def __init__(self, module, prefix, *args):
        self.module = module
        self.prefix = prefix
        self.num_module = 0
        for new_module in args:
            self.append(new_module)

    def append(self, new_module):
        if not isinstance(new_module, nn.Module):
            raise ValueError('Not a Module')
        else:
            self.module.add_module(self.prefix + str(self.num_module), new_module)
            self.num_module += 1

    def __len__(self):
        return self.num_module

    def __getitem__(self, i):
        if i < 0 or i >= self.num_module:
            raise IndexError('Out of bound')
        return getattr(self.module, self.prefix + str(i))


class mGRNDropout(torch.nn.Module):
    def __init__(self, rnn_hidden_size, num_classes, device, input_size_list,
                 size_of=1, dropouti = 0, dropoutw = 0, dropouto = 0,
                 keras_initialization = False, batch_first = False):
        '''

        :param rnn_hidden_size: unit dimension of marginal components.
        :param num_classes: dimension of final outputs
        :param batch_first: whether the first dimension of inputs is batch_size.
                    batch_first = True is not implemented.
        :param device: cpu or cuda
        :param input_size_list: the number of features in each marginal component.
                    The inputs should be sorted in the same sequence.
        :param size_of: unit dimension of the joint component is size_of*rnn_hidden_size
        :param dropouti: input dropout of both marginal and joint components.
        :param dropoutw: recurrent dropout of both marginal and joint components.
        :param dropouto: output dropout between joint layer and the ouput dense layer.
        :param keras_initialization: If True, use Keras default initialization.
                    If False, use Pytorch default initialization.
                    To produce comparable results as both ChannelwiseLSTM and LSTM-FCN papers
                    use Keras initialization.
        '''
        super(mGRNDropout, self).__init__()
        self.hidden_size = rnn_hidden_size
        self.batch_first = batch_first
        self.input_size_list = input_size_list
        self.marginal_list = ListModule(self, 'marginal_')
        self.device = device
        self.size_of = size_of
        self.dropoutw = dropoutw
        self.dropouti = dropouti
        self.keras_initialization = keras_initialization
        self.joint_size = int(size_of*self.hidden_size)
        for input_size in self.input_size_list:
            self.marginal_list.append(GRUDropoutLayer(input_size, rnn_hidden_size, self.dropoutw,
                                                      self.keras_initialization,
                                                      batch_first=batch_first,
                                                      dropouti=self.dropouti))
        self.input_size = sum(list(input_size_list)) # number of total input features
        self.n_group = len(input_size_list)
        self.joint = JointDropoutLayer(self.input_size, rnn_hidden_size, self.n_group,
                                       self.dropoutw, self.joint_size,
                                       self.keras_initialization,
                                       batch_first=batch_first, dropouti=self.dropouti)
        self.fc1 = torch.nn.Linear(self.joint_size, num_classes)
        if self.keras_initialization:
            for name, param in self.fc1.named_parameters():
                if 'b' in name:
                    nn.init.constant_(param.data, 0)
                elif 'w' in name:
                    nn.init.xavier_uniform_(param.data)
        self.dropout_output = torch.nn.Dropout(dropouto)
        self.param_num = self.get_param_num()

    def get_param_num(self):
        # number of trainable parameters excluding output dense layer.
        rnn_units = self.hidden_size
        n_group = self.n_group
        n_feature = self.input_size
        size_of = self.size_of
        n_para = (rnn_units * rnn_units * 3 + rnn_units * 3) * n_group + rnn_units * n_feature * (3 + size_of) \
                 + rnn_units * rnn_units * n_group * size_of + rnn_units * 2 * size_of + \
                 rnn_units * rnn_units * size_of * size_of
        return n_para

    def forward(self, x):
        # Set initial hidden and cell states
        if self.batch_first:
            batch_size = x.size(0)
        else:
            batch_size = x.size(1)
        h0 = torch.zeros(batch_size, self.hidden_size).to(self.device)
        input_index_beg = 0
        newgate_list = []
        out_list = []
        # print(self.input_size_list)
        for i, input_size in enumerate(self.input_size_list):
            gru = self.marginal_list[i]
            input_index_end = input_size + input_index_beg
            this_x = x[:, :, input_index_beg:input_index_end]
            this_out, this_newgate = gru(this_x, h0)
            input_index_beg = input_index_end
            newgate_list.append(this_newgate)
            out_list.append(this_out)

        newgate = torch.cat(newgate_list, dim=2)
        h0 = torch.zeros(batch_size, self.joint_size).to(self.device)
        out = self.joint(x, newgate, h0)
        if len(out.shape) == 3:
            if self.batch_first:
                out = out[:, -1, :]
            else:
                out = out[-1, :, :]
        out = self.dropout_output(out)
        out = self.fc1(out)
        return out