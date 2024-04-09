import jax.numpy as jnp


def batch_psnr(gen_frames, gt_frames):
    if gen_frames.ndim == 3:
        axis = (1, 2)
    elif gen_frames.ndim == 4:
        axis = (1, 2, 3)

    x = jnp.int32(gen_frames)
    y = jnp.int32(gt_frames)
    num_pixels = jnp.size(gen_frames[0]).astype(jnp.float32)

    mse = jnp.sum((x - y) ** 2, axis=axis, dtype=jnp.float32) / num_pixels
    psnr = 20 * jnp.log10(255) - 10 * jnp.log10(mse)

    return jnp.mean(psnr)

