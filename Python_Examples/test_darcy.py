from pathlib import Path
import sys
import matplotlib.pyplot as plt
import torch

from torch.utils.data import DataLoader, DistributedSampler
import wandb

from neuralop import H1Loss, LpLoss, Trainer, get_model
from neuralop.data.datasets import load_darcy_flow_small
from neuralop.data.transforms.data_processors import MGPatchingDataProcessor
from neuralop.training import setup, AdamW
from neuralop.mpu.comm import get_local_rank
from neuralop.utils import get_wandb_api_key, count_model_params


# Read the configuration
from zencfg import ConfigBase, cfg_from_commandline
import sys 
sys.path.insert(0, '../')
from config.darcy_config import Default


config = cfg_from_commandline(Default)
config = config.to_dict()

# Set-up distributed communication, if using
device, is_logger = setup(config)
print(device)
# Make sure we only print information when needed
config.verbose = config.verbose and is_logger

# Print config to screen
if config.verbose and is_logger:
    print(f"##### CONFIG #####\n")
    print(config)
    sys.stdout.flush()

# Loading the Darcy flow dataset
data_root = Path(config.data.folder).expanduser()
train_loader, test_loaders, data_processor = load_darcy_flow_small(
    data_root=data_root,
    n_train=config.data.n_train,
    batch_size=config.data.batch_size,
    test_resolutions=config.data.test_resolutions,
    n_tests=config.data.n_tests,
    test_batch_sizes=config.data.test_batch_sizes,
    encode_input=False,
    encode_output=False,
)


# load trained model
model = get_model(config)
model.load_state_dict(torch.load(
    "./ckpt/model_state_dict.pt",
    weights_only=False))
model = model.to(device)


# convert dataprocessor to an MGPatchingDataprocessor if patching levels > 0
if config.patching.levels > 0:
    data_processor = MGPatchingDataProcessor(model=model,
                                             in_normalizer=data_processor.in_normalizer,
                                             out_normalizer=data_processor.out_normalizer,
                                             padding_fraction=config.patching.padding,
                                             stitching=config.patching.stitching,
                                             levels=config.patching.levels,
                                             use_distributed=config.distributed.use_distributed,
                                             device=device)

# Reconfigure DataLoaders to use a DistributedSampler 
# if in distributed data parallel mode
if config.distributed.use_distributed:
    train_db = train_loader.dataset
    train_sampler = DistributedSampler(train_db, rank=get_local_rank())
    train_loader = DataLoader(dataset=train_db,
                              batch_size=config.data.batch_size,
                              sampler=train_sampler)
    for (res, loader), batch_size in zip(test_loaders.items(), config.data.test_batch_sizes):
        
        test_db = loader.dataset
        test_sampler = DistributedSampler(test_db, rank=get_local_rank())
        test_loaders[res] = DataLoader(dataset=test_db,
                              batch_size=batch_size,
                              shuffle=False,
                              sampler=test_sampler)


# Creating the losses
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)
if config.opt.training_loss == "l2":
    train_loss = l2loss
elif config.opt.training_loss == "h1":
    train_loss = h1loss
else:
    raise ValueError(
        f'Got training_loss={config.opt.training_loss} '
        f'but expected one of ["l2", "h1"]'
    )
eval_losses = {"h1": h1loss, "l2": l2loss}

if config.verbose and is_logger:
    print("\n### MODEL ###\n", model)
    print(f"\n### Beginning Testing...\n")
    sys.stdout.flush()

model.eval()
with torch.no_grad():
    # Take the first test loader (corresponding to the first test resolution)
    test_batch = next(iter(test_loaders[list(test_loaders.keys())[0]]))  
    inputs, targets = test_batch['x'], test_batch['y']

    inputs = inputs.to(device)
    targets = targets.to(device)

    outputs = model(inputs)

# Assume inputs, outputs, and targets have shape [batch_size, channels, height, width]

idx = 1  # Visualize only the first sample

# Move tensors to CPU and convert to numpy arrays
input_img = inputs[idx, 0].cpu().numpy()
output_img = outputs[idx, 0].cpu().numpy()
target_img = targets[idx, 0].cpu().numpy()

plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.title("Model Input", fontsize=16)
plt.imshow(input_img, cmap='viridis')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.title("Model Prediction", fontsize=16)
plt.imshow(output_img, cmap='viridis')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.title("Ground Truth", fontsize=16)
plt.imshow(target_img, cmap='viridis')
plt.colorbar()

plt.tight_layout()

save_path = "input_prediction_truth_{}.png".format(idx)
plt.savefig(save_path)
print("The test result is saved to " + save_path)
