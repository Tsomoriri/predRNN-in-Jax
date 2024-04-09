import numpy as np
import jax.numpy as jnp
import os
import shutil
import jax
import optax
import shutil
import argparse
from core.models.Jaxmodel_factory import Model
from core.utils.Jaxtsne import visualization
from core.utils.Jaxpreprocess import reshape_patch, reshape_patch_back
from core.utils.Jaxmetrics import batch_psnr


# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='PyTorch video prediction model - PredRNN')



# model
parser.add_argument('--model_name', type=str, default='predrnn')
parser.add_argument('--num_hidden', type=str, default='6,6,6,6')
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--patch_size', type=int, default=4)
parser.add_argument('--img_channel', type=int, default=1)
parser.add_argument('--img_width', type=int, default=40)
parser.add_argument('--total_length', type=int, default=5)
parser.add_argument('--input_length', type=int, default=4)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--reverse_scheduled_sampling', type=int, default=0)
parser.add_argument('--decouple_beta', type=float, default=0.1)
parser.add_argument('--save_dir', type=str, default='checkpoints')
parser.add_argument('--visual', type=int, default=0)
parser.add_argument('--visual_path', type=str, default='visual')
parser.add_argument('--filter_size', type=float, default=4)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--layer_norm', type=int, default=1)


args = parser.parse_args()
print(args)

# Load the data
movies = np.load('radar_movies.npy')

# Split the data into input sequences and target frames
x = movies[:, :, :, :4]  # Input sequences of length 4
y = movies[:, :, :, 4:5]  # Target frames of length 1
# Split the data into train and test sets
train_ratio = 0.8  # Adjust the ratio as needed
train_size = int(len(movies) * train_ratio)

train_movies = movies[:train_size]
test_movies = movies[train_size:]

train_x = train_movies[:, :, :, :4]
train_y = train_movies[:, :, :, 4:5]
test_x = test_movies[:, :, :, :4]
test_y = test_movies[:, :, :, 4:5]

# Create an instance of the Config class
configs = args

# Create an instance of the Model class
model = Model(configs)

# Train the model
num_epochs = 10  # Number of training epochs
for epoch in range(num_epochs):
    for i in range(0, len(x), configs.batch_size):
        batch_x = x[i:i + configs.batch_size]
        batch_y = y[i:i + configs.batch_size]

        # Prepare input frames and mask
        input_frames = np.zeros((configs.batch_size, configs.total_length, 40, 40, 1))
        input_frames[:, :4, :, :, 0] = batch_x
        input_frames[:, 4:5, :, :, 0] = batch_y

        mask = np.zeros((configs.batch_size, configs.total_length, 40, 40, 1))
        mask[:, 4:5, :, :, :] = 1  # Set mask to 1 for target frames

        # Train the model
        loss = model.train(input_frames, mask)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(x)}], Loss: {loss:.4f}')

    # Save the model checkpoint after each epoch
    model.save(epoch + 1)

# Evaluate the model on the test set
test_psnr = []
for i in range(0, len(test_x), configs.batch_size):
    batch_x = test_x[i:i + configs.batch_size]
    batch_y = test_y[i:i + configs.batch_size]

    # Prepare input frames and mask
    input_frames = np.zeros((configs.batch_size, configs.total_length, 40, 40, 1))
    input_frames[:, :4, :, :, 0] = batch_x
    input_frames[:, 4:5, :, :, 0] = batch_y

    mask = np.zeros((configs.batch_size, configs.total_length, 40, 40, 1))
    mask[:, 4:5, :, :, :] = 1  # Set mask to 1 for target frames

    # Generate predictions
    pred_frames = model.test(input_frames, mask)

    # Calculate PSNR
    psnr = batch_psnr(pred_frames[:, 4:5], batch_y)
    test_psnr.append(psnr)

print(f'Average Test PSNR: {np.mean(test_psnr):.4f}')

# Visualize the predictions using t-SNE
num_samples = 10  # Number of samples to visualize
vis_x = test_x[:num_samples]
vis_y = test_y[:num_samples]

# Prepare input frames and mask
input_frames = np.zeros((num_samples, configs.total_length, 40, 40, 1))
input_frames[:, :4, :, :, 0] = vis_x
input_frames[:, 4:5, :, :, 0] = vis_y

mask = np.zeros((num_samples, configs.total_length, 40, 40, 1))
mask[:, 4:5, :, :, :] = 1  # Set mask to 1 for target frames

# Generate predictions
pred_frames = model.test(input_frames, mask)

# Reshape frames to patches
vis_x_patch = reshape_patch(input_frames[:, :4], configs.patch_size)
vis_y_patch = reshape_patch(input_frames[:, 4:5], configs.patch_size)
pred_patch = reshape_patch(pred_frames[:, 4:5], configs.patch_size)

# Visualize using t-SNE
visualization(configs.total_length, model.num_layers, vis_x_patch, vis_y_patch, pred_patch, 'tsne_viz')