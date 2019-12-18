# ref: https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py
import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: int or (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.padding = self.kernel_size[0] // 2, self.kernel_size[1] // 2
        self.bias = bias

        self.conv_x = nn.Conv2d(in_channels=self.input_dim, out_channels=4 * self.hidden_dim,
                                kernel_size=self.kernel_size, padding=self.padding, bias=self.bias)
        self.conv_h = nn.Conv2d(in_channels=self.hidden_dim, out_channels=4 * self.hidden_dim,
                                kernel_size=self.kernel_size, padding=self.padding, bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        cx = self.conv_x(input_tensor)
        ch = self.conv_h(h_cur)

        cx_i, cx_f, cx_o, cx_g = torch.split(cx, self.hidden_dim, dim=1)
        ch_i, ch_f, ch_o, ch_g = torch.split(ch, self.hidden_dim, dim=1)
        i = torch.sigmoid(cx_i + ch_i)
        f = torch.sigmoid(cx_f + ch_f)
        o = torch.sigmoid(cx_o + ch_o)
        g = torch.tanh(cx_g + ch_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_dim, self.height, self.width),
                torch.zeros(batch_size, self.hidden_dim, self.height, self.width))


class ConvLSTM(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        if not len(hidden_dim) == len(kernel_size) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        layer_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            layer_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                           input_dim=cur_input_dim,
                                           hidden_dim=self.hidden_dim[i],
                                           kernel_size=self.kernel_size[i],
                                           bias=self.bias))

        self.layer_list = nn.ModuleList(layer_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: 5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: None

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))   # len = num_layers

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        device = input_tensor.device

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx][0].to(device), hidden_state[layer_idx][1].to(device)
            output_inner = []

            for t in range(seq_len):
                h, c = self.layer_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)     # (b, t, c, h, w)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:                          # return the last layer output
            layer_output_list = layer_output_list[-1]
            last_state_list = last_state_list[-1]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.layer_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


if __name__ == '__main__':
    # gradient check
    convlstm = ConvLSTM(input_size=(64, 32), input_dim=512, hidden_dim=[128, 64, 64, 32, 32],
                        kernel_size=[3, 3, 3, 3, 3], num_layers=5, batch_first=True)

    convlstm = convlstm.cuda()

    loss_fn = torch.nn.MSELoss()

    input = torch.randn(1, 5, 512, 64, 32).cuda()   # (b, t, c, h, w)
    target = torch.randn(1, 5, 32, 64, 32).cuda()

    output, _ = convlstm(input)                     # output: (b, t, c, h, w)
    output = output.double()

    res = torch.autograd.gradcheck(loss_fn, (output, target.double()), eps=1e-6, raise_exception=True)
    print(res)
