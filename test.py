import os
import torch
from torch.utils.data import DataLoader

from vae import VAE
from utils import save_output, save_both, view_one
from VoxelDataset import RotatedVoxelDataset
import torchvision
import torchvision.datasets as datasets
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
checkpoint = torch.load("/home/anurag/codes/generative_3d/original_v2/op2/best_model.pt")
model.load_state_dict(checkpoint)
model.eval()

test_voxels = RotatedVoxelDataset(train=False)
train_dataloader = DataLoader(test_voxels, batch_size=1, shuffle=True)

save_path = "/home/anurag/codes/generative_3d/original_v2/for_paper"
if not os.path.exists(save_path):
    os.makedirs(save_path)

for i, data in enumerate(train_dataloader):
    sample = data[0].to(device)

    reconstructions,_,_ = model(sample)
    level = 0
    reconstructions[reconstructions > level] = 1
    reconstructions[reconstructions < level] = 0

    reconstructions = reconstructions.detach().cpu()
    # save_output(reconstructions[0][0], 32, 'reconstructions', i)
    label = "Digit: {} x: {:.2f} y: {:.2f} z: {:.2f}".format(int(data[1].item()), 
                                                        data[2].item(), 
                                                        data[3].item(), 
                                                        data[4].item())
    print("Saving", os.path.join(save_path, "reconstruction_{}.png".format(i)))
    save_both(os.path.join(save_path, "reconstruction_{}.png".format(i)), label, data[0].squeeze().squeeze().detach().cpu().numpy(),reconstructions.squeeze().squeeze().detach().cpu())
    print("Saving", os.path.join(save_path, "reconstruction_{}.png".format(i)))

    if i != 0 and i % 10 == 0:
        break

view_one(label, data[0].squeeze().squeeze().detach().cpu().numpy(),reconstructions.squeeze().squeeze().detach().cpu())

