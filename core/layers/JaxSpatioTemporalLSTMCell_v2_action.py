import equinox as eqx
import jax.numpy as jnp
import jax

class ConvLayer(eqx.Module):
    conv: eqx.nn.Conv2d
    layer_norm: eqx.nn.LayerNorm
    use_layer_norm: bool

    def __init__(self, in_channel, out_channel, filter_size, stride, padding, use_layer_norm=True):
        self.conv = eqx.nn.Conv2d(in_channel, out_channel, kernel_size=filter_size, stride=stride, padding=padding,
                                  use_bias=True)
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = eqx.nn.LayerNorm((out_channel,))

    def __call__(self, x):
        x = self.conv(x)
        if self.use_layer_norm:
            x = self.layer_norm(x)
        return x


class SpatioTemporalLSTMCell(eqx.Module):
    conv_x: ConvLayer
    conv_h: ConvLayer
    conv_a: ConvLayer
    conv_m: ConvLayer
    conv_o: ConvLayer
    conv_last: eqx.nn.Conv2d
    num_hidden: int
    _forget_bias: float

    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm):
        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0

        self.conv_x = ConvLayer(in_channel, num_hidden * 7, filter_size, stride, self.padding,
                                use_layer_norm=layer_norm)
        self.conv_h = ConvLayer(num_hidden, num_hidden * 4, filter_size, stride, self.padding,
                                use_layer_norm=layer_norm)
        self.conv_a = ConvLayer(num_hidden, num_hidden * 4, filter_size, stride, self.padding,
                                use_layer_norm=layer_norm)
        self.conv_m = ConvLayer(num_hidden, num_hidden * 3, filter_size, stride, self.padding,
                                use_layer_norm=layer_norm)
        self.conv_o = ConvLayer(num_hidden * 2, num_hidden, filter_size, stride, self.padding,
                                use_layer_norm=layer_norm)
        self.conv_last = eqx.nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, use_bias=True)

    def __call__(self, x_t, h_t, c_t, m_t, a_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        a_concat = self.conv_a(a_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = jnp.split(x_concat, self.num_hidden, axis=1)
        i_h, f_h, g_h, o_h = jnp.split(h_concat * a_concat, self.num_hidden, axis=1)
        i_m, f_m, g_m = jnp.split(m_concat, self.num_hidden, axis=1)

        i_t = jax.nn.sigmoid(i_x + i_h)
        f_t = jax.nn.sigmoid(f_x + f_h + self._forget_bias)
        g_t = jnp.tanh(g_x + g_h)

        delta_c = i_t * g_t
        c_new = f_t * c_t + delta_c

        i_t_prime = jax.nn.sigmoid(i_x_prime + i_m)
        f_t_prime = jax.nn.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = jnp.tanh(g_x_prime + g_m)

        delta_m = i_t_prime * g_t_prime
        m_new = f_t_prime * m_t + delta_m

        mem = jnp.concatenate((c_new, m_new), axis=1)
        o_t = jax.nn.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * jnp.tanh(self.conv_last(mem))

        return h_new, c_new, m_new, delta_c, delta_m