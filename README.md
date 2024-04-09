# Implementation of ![predRNN v2](https://github.com/thuml/predrnn-pytorch/blob/master/core/layers/SpatioTemporalLSTMCell_v2_action.py) in Jax. 

## Motivations:

Jax has features such as autoparallelism and just in time compilation of graphs.The goal for this project is to understand deep learning framework jax by porting the official implementation written in Pytorch and utilise the native Jax features. The neural network library equinox and optimisers from optax are used in this project.

## Usage
The train.py is hard coded to handle a radar movie of format [:,:,:,:,].npy.
```bash
python -u train.py --model_name predrnn --num_hidden 16,16,16,16,16 --num_layers 5 --patch_size 4 --img_channel 1 --img_width 40 --total_length 5 --input_length 4 --lr 0.001 --batch_size 16 --reverse_scheduled_sampling 0 --decouple_beta 0.1 --save_dir checkpoints --visual 1

```

## A table mapping Pytorch functions to Jax 

| PyTorch Function                    | JAX Function                                     |
|-------------------------------------|--------------------------------------------------|
| `torch.tensor`                      | `jax.numpy.array`                                |
| `torch.zeros`                       | `jax.numpy.zeros`                                |
| `torch.ones`                        | `jax.numpy.ones`                                 |
| `torch.randn`                       | `jax.random.normal`                              |
| `torch.cat`                         | `jax.numpy.concatenate`                          |
| `torch.stack`                       | `jax.numpy.stack`                                |
| `torch.reshape`                     | `jax.numpy.reshape`                              |
| `torch.transpose`                   | `jax.numpy.transpose`                            |
| `torch.sum`                         | `jax.numpy.sum`                                  |
| `torch.mean`                        | `jax.numpy.mean`                                 |
| `torch.max`                         | `jax.numpy.max`                                  |
| `torch.min`                         | `jax.numpy.min`                                  |
| `torch.argmax`                      | `jax.numpy.argmax`                               |
| `torch.argmin`                      | `jax.numpy.argmin`                               |
| `torch.exp`                         | `jax.numpy.exp`                                  |
| `torch.log`                         | `jax.numpy.log`                                  |
| `torch.sin`                         | `jax.numpy.sin`                                  |
| `torch.cos`                         | `jax.numpy.cos`                                  |
| `torch.tanh`                        | `jax.numpy.tanh`                                 |
| `torch.sigmoid`                     | `jax.nn.sigmoid`                                 |
| `torch.relu`                        | `jax.nn.relu`                                    |
| `torch.softmax`                     | `jax.nn.softmax`                                 |
| `torch.nn.Conv2d`                   | `equinox.nn.Conv2d`                              |
| `torch.nn.ConvTranspose2d`          | `equinox.nn.ConvTranspose2d`                     |
| `torch.nn.Linear`                   | `equinox.nn.Linear`                              |
| `torch.nn.LayerNorm`                | `equinox.nn.LayerNorm`                           |
| `torch.nn.BatchNorm2d`              | `equinox.nn.BatchNorm2d`                         |
| `torch.nn.Dropout`                  | `equinox.nn.Dropout`                             |
| `torch.nn.Embedding`                | `equinox.nn.Embedding`                           |
| `torch.optim.Adam`                  | `optax.adam`                                     |
| `torch.optim.SGD`                   | `optax.sgd`                                      |
| `torch.optim.RMSprop`               | `optax.rmsprop`                                  |
| `torch.nn.Module`                   | `equinox.Module`                                 |
| `torch.no_grad`                     | `jax.disable_grad`                               |
| `torch.grad`                        | `jax.grad`                                       |
| `torch.autograd.grad`               | `jax.grad`                                       |
| `torch.nn.functional.mse_loss`      | `jax.numpy.mean((pred - target) ** 2)`           |
| `torch.nn.functional.cross_entropy` | `optax.softmax_cross_entropy`                    |


Original paper by: 
```
@inproceedings{wang2017predrnn,
  title={{PredRNN}: Recurrent Neural Networks for Predictive Learning Using Spatiotemporal {LSTM}s},
  author={Wang, Yunbo and Long, Mingsheng and Wang, Jianmin and Gao, Zhifeng and Yu, Philip S},
  booktitle={Advances in Neural Information Processing Systems},
  pages={879--888},
  year={2017}
}

@misc{wang2021predrnn,
      title={{PredRNN}: A Recurrent Neural Network for Spatiotemporal Predictive Learning}, 
      author={Wang, Yunbo and Wu, Haixu and Zhang, Jianjin and Gao, Zhifeng and Wang, Jianmin and Yu, Philip S and Long, Mingsheng},
      year={2021},
      eprint={2103.09504},
      archivePrefix={arXiv},
}
```
