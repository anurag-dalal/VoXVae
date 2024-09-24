import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import RegularGridInterpolator

class RotatedVoxelDataset(Dataset):
    def __init__(self, seed=0, train=True):
        if train:
            self.data = datasets.MNIST(root='./data', train=True, download=True, transform=None)
        else:
            self.data = datasets.MNIST(root='./data', train=False, download=True, transform=None)
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        self.voxel_dataset = []
        for d in self.data:
            voxel = self.convert_to_3d_voxel(d[0])
            x_angle = np.pi*np.random.uniform(low=0.0, high=1.0, size=None)
            y_angle = np.pi*np.random.uniform(low=0.0, high=1.0, size=None)
            z_angle = np.pi*np.random.uniform(low=0.0, high=1.0, size=None)
            voxel = self.rotate_voxels(voxel, x_angle, y_angle, z_angle)
            voxel = voxel/255
            
            voxel = np.expand_dims(voxel, axis=0)
            self.voxel_dataset.append([voxel.astype(np.float32), d[1], x_angle, y_angle, z_angle])
            # self.voxel_dataset.append(voxel)


    def __len__(self):
        # return 200
        return len(self.voxel_dataset)

    def __getitem__(self, idx):
        return self.voxel_dataset[idx][0], self.voxel_dataset[idx][1], self.voxel_dataset[idx][2], self.voxel_dataset[idx][3], self.voxel_dataset[idx][4]
        # return self.voxel_dataset[idx]
    
    def convert_to_3d_voxel(self, image):
        voxel = np.zeros((28, 28, 28), dtype = np.uint8)
        voxel[:,28//2,:] = image
        voxel = np.pad(voxel, pad_width=((2, 2), (2, 2), (2, 2)), mode='constant', constant_values=0)
        return voxel
    
    def rotate_voxels(self, voxel, x_angle, y_angle, z_angle):
        r = R.from_euler('xyz',angles=[x_angle,y_angle,z_angle],degrees=False)
        M = r.as_matrix()
        trans_mat_inv = np.linalg.inv(M)
        
        Nz, Ny, Nx = voxel.shape
        x_center = Nx//2
        y_center = Ny//2
        z_center = Nz//2

        x = np.linspace(0, Nx - 1, Nx)
        y = np.linspace(0, Ny - 1, Ny)
        z = np.linspace(0, Nz - 1, Nz)
        zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
        coor = np.array([xx - x_center, yy - y_center, zz - z_center])

        coor_prime = np.tensordot(trans_mat_inv, coor, axes=((1), (0)))
        xx_prime = coor_prime[0] + x_center
        yy_prime = coor_prime[1] + y_center
        zz_prime = coor_prime[2] + z_center

        x_valid1 = xx_prime>=0
        x_valid2 = xx_prime<=Nx-1
        y_valid1 = yy_prime>=0
        y_valid2 = yy_prime<=Ny-1
        z_valid1 = zz_prime>=0
        z_valid2 = zz_prime<=Nz-1
        valid_voxel = x_valid1 * x_valid2 * y_valid1 * y_valid2 * z_valid1 * z_valid2
        z_valid_idx, y_valid_idx, x_valid_idx = np.where(valid_voxel > 0)

        image_transformed = np.zeros((Nz, Ny, Nx))
        data_w_coor = RegularGridInterpolator((z, y, x), voxel, method="nearest")
        interp_points = np.array([zz_prime[z_valid_idx, y_valid_idx, x_valid_idx],
                yy_prime[z_valid_idx, y_valid_idx, x_valid_idx],
                xx_prime[z_valid_idx, y_valid_idx, x_valid_idx]]).T
        interp_result = data_w_coor(interp_points)
        image_transformed[z_valid_idx, y_valid_idx, x_valid_idx] = interp_result
        return image_transformed

    def save_n_random(self, n, fname):
        # Get random indices
        indices = random.sample(range(len(self.voxel_dataset)), n)
        fig = plt.figure(figsize=(5*n, 5))
        for i in range(5):
            ax = fig.add_subplot(2, 5, i+1, projection='3d')
            print(self.voxel_dataset[indices[i]][0][0,:,:,:].shape)
            ax.voxels(self.voxel_dataset[indices[i]][0][0,:,:,:], edgecolor='k')
            ax.set_title("Label: %d %.2f %.2f %.2f" %(self.voxel_dataset[indices[i]][1], self.voxel_dataset[indices[i]][2], self.voxel_dataset[indices[i]][3], self.voxel_dataset[indices[i]][4]))
        plt.savefig(fname)

# Defining main function
def main():
    #train_voxels = RotatedVoxelDataset(train=True)
    test_voxels = RotatedVoxelDataset(train=False)
    #train_voxels.save_n_random(5, "train.png")
    test_voxels.save_n_random(5, "test.png")


# Using the special variable 
# __name__
if __name__=="__main__":
    main()