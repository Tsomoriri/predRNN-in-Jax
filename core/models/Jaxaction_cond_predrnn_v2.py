import equinox as eqx
import jax
import jax.numpy as jnp
from typing import List, Tuple
from core.layers.JaxSpatioTemporalLSTMCell_v2_action  import SpatioTemporalLSTMCell


class ConvLayer(eqx.Module):
    conv: eqx.nn.Conv2d

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        self.conv = eqx.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, use_bias=bias)

    def __call__(self, x):
        return self.conv(x)


class DeconvLayer(eqx.Module):
    deconv: eqx.nn.ConvTranspose2d

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        self.deconv = eqx.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, use_bias=bias)

    def __call__(self, x, output_size=None):
        return self.deconv(x, output_size=output_size)


class RNN(eqx.Module):
    conv_on_input: int
    res_on_conv: int
    patch_height: int
    patch_width: int
    patch_ch: int
    action_ch: int
    rnn_height: int
    rnn_width: int
    num_layers: int
    num_hidden: List[int]
    beta: float
    cell_list: List[SpatioTemporalLSTMCell]
    conv_input1: ConvLayer
    conv_input2: ConvLayer
    action_conv_input1: ConvLayer
    action_conv_input2: ConvLayer
    deconv_output1: DeconvLayer
    deconv_output2: DeconvLayer
    conv_last: eqx.nn.Conv2d
    adapter: eqx.nn.Conv2d

    def __init__(self, num_layers, num_hidden, configs):
        super().__init__()

        self.configs = configs
        self.conv_on_input = self.configs.conv_on_input
        self.res_on_conv = self.configs.res_on_conv
        self.patch_height = configs.img_width // configs.patch_size
        self.patch_width = configs.img_width // configs.patch_size
        self.patch_ch = configs.img_channel * (configs.patch_size ** 2)
        self.action_ch = configs.num_action_ch
        self.rnn_height = self.patch_height
        self.rnn_width = self.patch_width

        if self.configs.conv_on_input == 1:
            self.rnn_height = self.patch_height // 4
            self.rnn_width = self.patch_width // 4
            self.conv_input1 = ConvLayer(self.patch_ch, num_hidden[0] // 2, configs.filter_size, stride=2,
                                         padding=configs.filter_size // 2, bias=False)
            self.conv_input2 = ConvLayer(num_hidden[0] // 2, num_hidden[0], configs.filter_size, stride=2,
                                         padding=configs.filter_size // 2, bias=False)
            self.action_conv_input1 = ConvLayer(self.action_ch, num_hidden[0] // 2, configs.filter_size, stride=2,
                                                padding=configs.filter_size // 2, bias=False)
            self.action_conv_input2 = ConvLayer(num_hidden[0] // 2, num_hidden[0], configs.filter_size, stride=2,
                                                padding=configs.filter_size // 2, bias=False)
            self.deconv_output1 = DeconvLayer(num_hidden[num_layers - 1], num_hidden[num_layers - 1] // 2,
                                              configs.filter_size, stride=2, padding=configs.filter_size // 2, bias=False)
            self.deconv_output2 = DeconvLayer(num_hidden[num_layers - 1] // 2, self.patch_ch, configs.filter_size,
                                              stride=2, padding=configs.filter_size // 2, bias=False)

        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []
        self.beta = configs.decouple_beta

        for i in range(num_layers):
            if i == 0:
                in_channel = self.patch_ch + self.action_ch if self.configs.conv_on_input == 0 else num_hidden[0]
            else:
                in_channel = num_hidden[i - 1]
            cell_list.append(SpatioTemporalLSTMCell(in_channel, num_hidden[i], self.rnn_width,
                                                    configs.filter_size, configs.stride, configs.layer_norm))
        self.cell_list = cell_list

        if self.configs.conv_on_input == 0:
            self.conv_last = eqx.nn.Conv2d(num_hidden[num_layers - 1], self.patch_ch + self.action_ch, 1, stride=1,
                                           padding=0, use_bias=False)
        self.adapter = eqx.nn.Conv2d(num_hidden[num_layers - 1], num_hidden[num_layers - 1], 1, stride=1, padding=0,
                                     use_bias=False)

    def __call__(self, all_frames, mask_true) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = jnp.transpose(all_frames, (0, 1, 4, 2, 3))
        input_frames = frames[:, :, :self.patch_ch, :, :]
        input_actions = frames[:, :, self.patch_ch:, :, :]
        mask_true = jnp.transpose(mask_true, (0, 1, 4, 2, 3))

        next_frames = []
        h_t = []
        c_t = []
        delta_c_list = []
        delta_m_list = []

        for i in range(self.num_layers):
            zeros = jnp.zeros([self.configs.batch_size, self.num_hidden[i], self.rnn_height, self.rnn_width])
            h_t.append(zeros)
            c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)

        decouple_loss = []
        memory = jnp.zeros([self.configs.batch_size, self.num_hidden[0], self.rnn_height, self.rnn_width])

        for t in range(self.configs.total_length - 1):
            if t == 0:
                net = input_frames[:, t]
            else:
                net = mask_true[:, t - 1] * input_frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            action = input_actions[:, t]

            if self.conv_on_input == 1:
                net_shape1 = net.shape
                net = self.conv_input1(net)
                if self.res_on_conv == 1:
                    input_net1 = net
                net_shape2 = net.shape
                net = self.conv_input2(net)
                if self.res_on_conv == 1:
                    input_net2 = net
                action = self.action_conv_input1(action)
                action = self.action_conv_input2(action)

            h_t[0], c_t[0], memory, delta_c, delta_m = self.cell_list[0](net, h_t[0], c_t[0], memory, action)
            delta_c_list[0] = jax.nn.normalize(self.adapter(delta_c).reshape(delta_c.shape[0], delta_c.shape[1], -1), axis=2)
            delta_m_list[0] = jax.nn.normalize(self.adapter(delta_m).reshape(delta_m.shape[0], delta_m.shape[1], -1), axis=2)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory, delta_c, delta_m = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory, action)
                delta_c_list[i] = jax.nn.normalize(self.adapter(delta_c).reshape(delta_c.shape[0], delta_c.shape[1], -1), axis=2)
                delta_m_list[i] = jax.nn.normalize(self.adapter(delta_m).reshape(delta_m.shape[0], delta_m.shape[1], -1), axis=2)

            for i in range(0, self.num_layers):
                cosine_sim = jnp.sum(delta_c_list[i] * delta_m_list[i], axis=2) / (
                            jnp.linalg.norm(delta_c_list[i], axis=2) * jnp.linalg.norm(delta_m_list[i], axis=2))
                decouple_loss.append(jnp.mean(jnp.abs(cosine_sim)))
            if self.conv_on_input == 1:
                if self.res_on_conv == 1:
                    x_gen = self.deconv_output1(h_t[self.num_layers - 1] + input_net2, output_size=net_shape2)
                    x_gen = self.deconv_output2(x_gen + input_net1, output_size=net_shape1)
                else:
                    x_gen = self.deconv_output1(h_t[self.num_layers - 1], output_size=net_shape2)
                    x_gen = self.deconv_output2(x_gen, output_size=net_shape1)
            else:
                x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        decouple_loss = jnp.mean(jnp.stack(decouple_loss, axis=0))
        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = jnp.transpose(jnp.stack(next_frames, axis=0), (1, 0, 3, 4, 2))
        mse_loss = jnp.mean((next_frames - all_frames[:, 1:, :, :, :next_frames.shape[4]]) ** 2)
        loss = mse_loss + self.beta * decouple_loss
        next_frames = next_frames[:, :, :, :, :self.patch_ch]
        return next_frames, loss