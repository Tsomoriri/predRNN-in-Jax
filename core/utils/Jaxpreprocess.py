import jax.numpy as jnp

def reshape_patch(img_tensor, patch_size):
    assert 5 == img_tensor.ndim
    batch_size = jnp.shape(img_tensor)[0]
    seq_length = jnp.shape(img_tensor)[1]
    img_height = jnp.shape(img_tensor)[2]
    img_width = jnp.shape(img_tensor)[3]
    num_channels = jnp.shape(img_tensor)[4]
    a = jnp.reshape(img_tensor, [batch_size, seq_length,
                                 img_height//patch_size, patch_size,
                                 img_width//patch_size, patch_size,
                                 num_channels])
    b = jnp.transpose(a, [0,1,2,4,3,5,6])
    patch_tensor = jnp.reshape(b, [batch_size, seq_length,
                                   img_height//patch_size,
                                   img_width//patch_size,
                                   patch_size*patch_size*num_channels])
    return patch_tensor

def reshape_patch_back(patch_tensor, patch_size):
    assert 5 == patch_tensor.ndim
    batch_size = jnp.shape(patch_tensor)[0]
    seq_length = jnp.shape(patch_tensor)[1]
    patch_height = jnp.shape(patch_tensor)[2]
    patch_width = jnp.shape(patch_tensor)[3]
    channels = jnp.shape(patch_tensor)[4]
    img_channels = channels // (patch_size*patch_size)
    a = jnp.reshape(patch_tensor, [batch_size, seq_length,
                                   patch_height, patch_width,
                                   patch_size, patch_size,
                                   img_channels])
    b = jnp.transpose(a, [0,1,2,4,3,5,6])
    img_tensor = jnp.reshape(b, [batch_size, seq_length,
                                 patch_height * patch_size,
                                 patch_width * patch_size,
                                 img_channels])
    return img_tensor