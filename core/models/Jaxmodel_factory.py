import os
import jax
import jax.numpy as jnp
import optax
from flax import serialization

from core.models import Jaxpredrnn_v2, Jaxaction_cond_predrnn_v2

class Model(object):
    def __init__(self, configs):
        self.configs = configs
        self.num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        self.num_layers = len(self.num_hidden)

        networks_map = {
            'predrnn': Jaxpredrnn_v2.RNN,
            'action_cond_predrnn_v2': Jaxaction_cond_predrnn_v2.RNN,
        }

        if configs.model_name in networks_map:
            Network = networks_map[configs.model_name]
            self.network = Network(self.num_layers, self.num_hidden, configs)
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)

        self.optimizer = optax.adam(learning_rate=configs.lr)
        self.optimizer_state = self.optimizer.init(self.network.state)

    def save(self, itr):
        stats = {}
        stats['net_param'] = serialization.to_state_dict(self.network.state)
        checkpoint_path = os.path.join(self.configs.save_dir, 'model.ckpt' + '-' + str(itr))
        with open(checkpoint_path, 'wb') as f:
            f.write(serialization.to_bytes(stats))
        print("save model to %s" % checkpoint_path)

    def load(self, checkpoint_path):
        print('load model:', checkpoint_path)
        with open(checkpoint_path, 'rb') as f:
            stats = serialization.from_bytes(f.read())
        self.network.state = serialization.from_state_dict(self.network.state, stats['net_param'])

    def train(self, frames, mask):
        frames_tensor = jnp.array(frames)
        mask_tensor = jnp.array(mask)

        def loss_fn(params):
            self.network.state = params
            next_frames, loss = self.network(frames_tensor, mask_tensor)
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(self.network.state)
        updates, self.optimizer_state = self.optimizer.update(grads, self.optimizer_state)
        self.network.state = optax.apply_updates(self.network.state, updates)

        return loss.item()

    def test(self, frames, mask):
        frames_tensor = jnp.array(frames)
        mask_tensor = jnp.array(mask)

        next_frames, _ = self.network(frames_tensor, mask_tensor)
        return next_frames

