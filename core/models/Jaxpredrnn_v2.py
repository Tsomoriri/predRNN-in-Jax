import jax
import jax.numpy as jnp
import equinox as eqx
from typing import List, Tuple
from core.layers.JaxSpatioTemporalLSTMCell_v2 import SpatioTemporalLSTMCell
from core.utils.Jaxtsne import visualization


class RNN(eqx.Module):
    frame_channel: int
    num_layers: int
    num_hidden: List[int]
    cell_list: List[SpatioTemporalLSTMCell]
    conv_last: eqx.nn.Conv2d
    adapter: eqx.nn.Conv2d
    configs: any
    visual: int
    visual_path: str

    def __init__(self, num_layers, num_hidden, configs):
        super().__init__()

        self.configs = configs
        self.visual = self.configs.visual
        self.visual_path = self.configs.visual_path

        self.frame_channel = configs.patch_size * configs.patch_size * configs.img_channel
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        width = configs.img_width // configs.patch_size

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, num_hidden[i], width, configs.filter_size,
                                       )
            )
        self.cell_list = cell_list
        self.conv_last = eqx.nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel, kernel_size=1, stride=1, padding=0,
                                       use_bias=False)
        # shared adapter
        adapter_num_hidden = num_hidden[0]
        self.adapter = eqx.nn.Conv2d(adapter_num_hidden, adapter_num_hidden, 1, stride=1, padding=0, use_bias=False)

    def __call__(self, frames_tensor, mask_true) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = jnp.transpose(frames_tensor, (0, 1, 4, 2, 3))
        mask_true = jnp.transpose(mask_true, (0, 1, 4, 2, 3))

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []
        delta_c_list = []
        delta_m_list = []
        if self.visual:
            delta_c_visual = []
            delta_m_visual = []

        decouple_loss = []

        for i in range(self.num_layers):
            zeros = jnp.zeros([batch, self.num_hidden[i], height, width])
            h_t.append(zeros)
            c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)

        memory = jnp.zeros([batch, self.num_hidden[0], height, width])

        for t in range(self.configs.total_length - 1):
            if self.configs.reverse_scheduled_sampling == 1:
                # reverse schedule sampling
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            else:
                # schedule sampling
                if t < self.configs.input_length:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                          (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            h_t[0], c_t[0], memory, delta_c, delta_m = self.cell_list[0](net, h_t[0], c_t[0], memory)
            delta_c_list[0] = jax.nn.normalize(self.adapter(delta_c).reshape(delta_c.shape[0], delta_c.shape[1], -1), axis=2)
            delta_m_list[0] = jax.nn.normalize(self.adapter(delta_m).reshape(delta_m.shape[0], delta_m.shape[1], -1), axis=2)
            if self.visual:
                delta_c_visual.append(delta_c.reshape(delta_c.shape[0], delta_c.shape[1], -1))
                delta_m_visual.append(delta_m.reshape(delta_m.shape[0], delta_m.shape[1], -1))

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory, delta_c, delta_m = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)
                delta_c_list[i] = jax.nn.normalize(self.adapter(delta_c).reshape(delta_c.shape[0], delta_c.shape[1], -1), axis=2)
                delta_m_list[i] = jax.nn.normalize(self.adapter(delta_m).reshape(delta_m.shape[0], delta_m.shape[1], -1), axis=2)
                if self.visual:
                    delta_c_visual.append(delta_c.reshape(delta_c.shape[0], delta_c.shape[1], -1))
                    delta_m_visual.append(delta_m.reshape(delta_m.shape[0], delta_m.shape[1], -1))

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)
            # decoupling loss
            for i in range(0, self.num_layers):
                cosine_sim = jnp.sum(delta_c_list[i] * delta_m_list[i], axis=2) / (jnp.linalg.norm(delta_c_list[i], axis=2) * jnp.linalg.norm(delta_m_list[i], axis=2))
                decouple_loss.append(jnp.mean(jnp.abs(cosine_sim)))

        if self.visual:
            # visualization of delta_c and delta_m
            delta_c_visual = jnp.stack(delta_c_visual, axis=0)
            delta_m_visual = jnp.stack(delta_m_visual, axis=0)
            visualization(self.configs.total_length, self.num_layers, delta_c_visual, delta_m_visual, self.visual_path)
            self.visual = 0

        decouple_loss = jnp.mean(jnp.stack(decouple_loss, axis=0))
        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = jnp.transpose(jnp.stack(next_frames, axis=0), (1, 0, 3, 4, 2))
        mse_loss = jnp.mean((next_frames - frames_tensor[:, 1:]) ** 2)
        loss = mse_loss + self.configs.decouple_beta * decouple_loss
        return next_frames, loss