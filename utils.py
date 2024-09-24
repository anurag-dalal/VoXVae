import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from tqdm import tqdm
import pandas as pd
import torch
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt


def save_output(output_arr, output_size, output_dir, file_idx):
    plot_out_arr = np.array([])
    with_border_arr = np.zeros([34, 34, 34])
    for x_i in range(0, output_size):
        for y_j in range(0, output_size):
            for z_k in range(0, output_size):
                plot_out_arr = np.append(plot_out_arr, output_arr[x_i, y_j, z_k])
                
    text_save = np.reshape(plot_out_arr, (output_size * output_size * output_size))
    np.savetxt(output_dir + '/volume' + str(file_idx) + '.txt', text_save)

    output_image = np.reshape(plot_out_arr, (output_size, output_size, output_size)).astype(np.float32)

    for x_i in range(0, output_size):
        for y_j in range(0, output_size):
            for z_k in range(0, output_size):
                with_border_arr[x_i + 1, y_j + 1, z_k + 1] = output_image[x_i, y_j, z_k]

    if not np.any(with_border_arr):
        verts, faces, normals, values = [], [], [], []
    else:
        verts, faces, normals, values = measure.marching_cubes(with_border_arr, level = 0.0, gradient_direction = 'descent')
        faces = faces + 1

    obj_save = open(output_dir + '/volume' + str(file_idx) + '.obj', 'w')
    for item in verts:
        obj_save.write('v {0} {1} {2}\n'.format(item[0], item[1], item[2]))
    for item in normals:
        obj_save.write('vn {0} {1} {2}\n'.format(-item[0], -item[1], -item[2]))
    for item in faces:
        obj_save.write('f {0}//{0} {1}//{1} {2}//{2}\n'.format(item[0], item[2], item[1]))
    obj_save.close()

    output_image = np.rot90(output_image)
    x, y, z = output_image.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(x, y, z, zdir = 'z', c = 'red')
    plt.savefig(output_dir + '/volume' + str(file_idx) + '.png')
    plt.close()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def save_both(fname, label, orig, pred):
    """
    Save two voxel grids side by side with a label.

    Parameters:
    - fname (str): The filename to save the image.
    - label (str or int): The label to display.
    - orig (numpy.ndarray): Original voxel grid of shape (32, 32, 32).
    - pred (numpy.ndarray): Predicted voxel grid of shape (32, 32, 32).
    """
    # Ensure that orig and pred are binary for better visualization
    orig_binary = orig > 0.5
    pred_binary = pred > 0.5

    # Create a figure with two 3D subplots
    fig = plt.figure(figsize=(12, 6))

    # Plot Original Voxel Grid
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.voxels(orig_binary, facecolors='blue', edgecolor='k', alpha=0.7)
    ax1.set_title('Original', fontsize=14)
    ax1.axis('off')  # Hide axes for better visualization

    # Plot Predicted Voxel Grid
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.voxels(pred_binary, facecolors='red', edgecolor='k', alpha=0.7)
    ax2.set_title('Predicted', fontsize=14)
    ax2.axis('off')  # Hide axes for better visualization

    # Add a main title with the label
    fig.suptitle(f'{label}', fontsize=16, y=0.02)

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the figure to the specified file
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory

def view_one(label, orig, pred):
    """
    Save two voxel grids side by side with a label.

    Parameters:
    - fname (str): The filename to save the image.
    - label (str or int): The label to display.
    - orig (numpy.ndarray): Original voxel grid of shape (32, 32, 32).
    - pred (numpy.ndarray): Predicted voxel grid of shape (32, 32, 32).
    """
    # Ensure that orig and pred are binary for better visualization
    orig_binary = orig > 0.5
    pred_binary = pred > 0.5

    # Create a figure with two 3D subplots
    fig = plt.figure(figsize=(12, 6))

    # Plot Original Voxel Grid
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.voxels(orig_binary, facecolors='blue', edgecolor='k', alpha=0.7)
    ax1.set_title('Original', fontsize=14)
    ax1.axis('off')  # Hide axes for better visualization

    # Plot Predicted Voxel Grid
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.voxels(pred_binary, facecolors='red', edgecolor='k', alpha=0.7)
    ax2.set_title('Predicted', fontsize=14)
    ax2.axis('off')  # Hide axes for better visualization

    # Add a main title with the label
    fig.suptitle(f'Label: {label}', fontsize=16)

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the figure to the specified file
    plt.show()
    plt.close(fig)  # Close the figure to free memory

def train_model(epoch_num, model, train_dataloader, device, optimizer, scheduler, beta, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    loss_df = pd.DataFrame(columns=['epoch', 'bce', 'kld', 'loss', 'l1', 'RMSE'])
    best_loss = float('inf')
    best_epoch = -1
    epoch_progress = tqdm(range(1, epoch_num + 1), desc='Training', unit='epoch')
    # Training Loop
    epoch_progress.set_postfix({'Best': f"{best_epoch}", 'Loss': f"{-1}"})
    for epoch, _ in enumerate(epoch_progress, 1):
        model.train()  # Set model to training mode
        running_loss = 0.0  # To accumulate loss over the epoch
        running_bce = 0.0
        running_kld = 0.0
        running_l1 = 0.0
        running_rmse = 0.0
        # Initialize tqdm progress bar
        # progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{epoch_num}", unit="batch")
        
        for i, data in enumerate(train_dataloader):
            inputs, labels, *angles = data  # Unpack data if more than inputs are present
            inputs = inputs.to(device)  # Move inputs to the specified device
            
            optimizer.zero_grad()  # Zero the gradients
            
            outputs, mu, sigma = model(inputs)  # Forward pass
            BCE, KLD, fp = model.loss(inputs, outputs, mu, sigma)  # Compute loss
            loss = BCE + beta*(KLD) + fp
            
            loss.backward()  # Backward pass
            optimizer.step()  # Update weigh1ts
            
            running_loss += loss.item()  # Accumulate loss
            running_bce += BCE.item()
            running_kld += KLD.item()

            # Compute L1 loss (Mean Absolute Error)
            l1_loss = F.l1_loss(outputs, inputs)
            
            # Compute RMSE (Root Mean Squared Error)
            rmse = ((outputs - inputs) ** 2).mean().sqrt()  # RMSE is the square root of MSE
            running_l1 += l1_loss.item()
            running_rmse += rmse.item()
            

        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(train_dataloader)
        running_bce = running_bce / len(train_dataloader)
        running_kld = running_kld / len(train_dataloader)
        running_l1 = running_l1 / len(train_dataloader)
        running_rmse = running_rmse / len(train_dataloader)
        loss_df.loc[len(loss_df)] = {
            'epoch': epoch,
            'bce': running_bce,
            'kld': running_kld,
            'loss': epoch_loss,
            'l1': running_l1,
            'rmse': running_rmse
        }
        # Check if this epoch's loss is the best so far
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))  # Save the best model
            best_epoch = epoch
        epoch_progress.set_postfix({'Best': f"{best_epoch}", 'Loss': f"{epoch_loss:.4f}"})
        # Step the scheduler
        scheduler.step()
        
        # Optionally, print epoch summary (if you want additional logging)
        # print(f"Epoch {epoch}/{epoch_num} completed. Average Loss: {epoch_loss:.4f}")

    loss_df.to_csv(os.path.join(save_dir, 'train_loss_data.csv'), index=False)
    print("Loss data saved as 'loss_data.csv'.")

    print('Finished Training')

    # Save the final model after all epochs
    torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pt"))
    print("Final model saved as 'final_model.pt'")

    # Plot and save the figure for BCE, KLD, and loss over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(loss_df['epoch'], loss_df['bce'], label='BCE Loss')
    plt.plot(loss_df['epoch'], loss_df['kld'], label='KLD Loss')
    plt.plot(loss_df['epoch'], loss_df['loss'], label='Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.legend()
    plt.grid(True)

    # Save the figure
    plt.savefig(os.path.join(save_dir, 'loss_plot.png'))

    # Plot L1 Loss over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(loss_df['epoch'], loss_df['l1'], label='L1 Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('L1 Loss')
    plt.title('L1 Loss vs Epoch')
    plt.grid(True)

    # Save the figure for L1 Loss
    plt.savefig(os.path.join(save_dir, 'l1_loss_plot.png'))

    # Plot RMSE over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(loss_df['epoch'], loss_df['RMSE'], label='RMSE', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('RMSE vs Epoch')
    plt.grid(True)

    # Save the figure for RMSE
    plt.savefig(os.path.join(save_dir, 'rmse_plot.png'))



def validate_model(model, dataloader, device, save_dir, fname):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    model = model.to(device)
    model.eval()
    
    total_l1_loss = 0.0
    total_accuracy = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_iou = 0.0
    total_rmse = 0.0
    total_batches = len(dataloader)
    
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs, labels, *angles = data  # Unpack data if more than inputs are present
            inputs = inputs.to(device)  # Move inputs to the specified device
            labels = labels.to(device)  # Move labels to the specified device
            
            # Forward pass
            outputs, mu, sigma = model(inputs)
            
            # Binarize outputs and labels (assuming threshold at 0.5 for binary voxels)
            outputs_binary = (outputs > 0.5).float()
            labels_binary = (inputs > 0.5).float()
            
            # RMSE
            total_rmse += ((outputs - inputs) ** 2).mean().sqrt().item()
            # L1 Loss
            l1_loss = F.l1_loss(outputs_binary, labels_binary)
            total_l1_loss += l1_loss.item()
            
            # Accuracy
            correct_predictions = torch.eq(outputs_binary, labels_binary).sum().item()
            accuracy = correct_predictions / torch.numel(labels_binary)
            total_accuracy += accuracy
            
            # Precision and Recall
            true_positive = (outputs_binary * labels_binary).sum().item()
            predicted_positive = outputs_binary.sum().item()
            actual_positive = labels_binary.sum().item()
            
            precision = true_positive / (predicted_positive + 1e-6)  # Avoid division by zero
            recall = true_positive / (actual_positive + 1e-6)        # Avoid division by zero
            
            total_precision += precision
            total_recall += recall
            
            # Intersection over Union (IoU)
            intersection = (outputs_binary * labels_binary).sum().item()
            union = outputs_binary.sum().item() + labels_binary.sum().item() - intersection
            iou = intersection / (union + 1e-6)  # Avoid division by zero
            
            total_iou += iou
            

    # Calculate average metrics over the entire validation set
    avg_l1_loss = total_l1_loss / total_batches
    avg_accuracy = total_accuracy / total_batches
    avg_precision = total_precision / total_batches
    avg_recall = total_recall / total_batches
    avg_iou = total_iou / total_batches
    avg_rmse = total_rmse / total_batches

    output_file = os.path.join(save_dir, fname)
    # Save the results to a txt file
    with open(output_file, 'w') as f:
        f.write(f"Average L1 Loss: {avg_l1_loss:.4f}\n")
        f.write(f"Average Accuracy: {avg_accuracy:.4f}\n")
        f.write(f"Average Precision: {avg_precision:.4f}\n")
        f.write(f"Average Recall: {avg_recall:.4f}\n")
        f.write(f"Average IoU: {avg_iou:.4f}\n")
        f.write(f"Average RMSE: {avg_rmse:.4f}\n")


    
    return avg_l1_loss, avg_accuracy, avg_precision, avg_recall, avg_iou

def save_from_dataloader(dataloader, model, device, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        

    for i, data in enumerate(dataloader):
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
        save_both(os.path.join(save_dir, "reconstruction_{}.png".format(i)), label, data[0].squeeze().squeeze().detach().cpu().numpy(),reconstructions.squeeze().squeeze().detach().cpu())
        print("Saved", os.path.join(save_dir, "reconstruction_{}.png".format(i)))

        if i != 0 and i % 10 == 0:
            break

