# Import libraries
from tqdm import tqdm
import torch
from skimage import io
import numpy as np
import pandas as pd
import networkx as nx 
import matplotlib.pyplot as plt

# Train function
def train_fn(m, dl, opt, loss_fn, device):
    
    """
    
    This function gets model, dataloader, optimizer, loss function, and gpu device;
    conducts one epoch of training;
    and return train loss for the epoch.
    
    Arguments:
    
        m       - model, torch model;
        dl      - train dataloader, torch dataloader object;
        opt     - optimizer, torch optimizer, object;
        loss_fn - loss function, torch loss function, object;
        device  - gpu device, str.
        
    Output:
    
        loss    - average loss value, float.
    
    """
    # Switch the model to train mode
    m.train()
    
    # Set initial total loss value
    total_loss = 0.
    
    # Go through the train dataloader
    for q, p, n in tqdm(dl):
        
        # Move query, positive, and negative images to gpu
        q, p, n = q.to(device),p.to(device),n.to(device)
        
        # Get feature maps for each image
        q_fms = m(q)
        p_fms = m(p)
        n_fms = m(n)

        # Compute the loss using the feature maps
        loss = loss_fn(q_fms, p_fms, n_fms)
        
        # Zero grad of the optimizer
        opt.zero_grad()
        
        # Conduct backpropagation
        loss.backward()
        
        # Update trainable parameters
        opt.step()
        
        # Add loss value of the mini-batch to the total loss 
        total_loss += loss.item()
        
    # Return average loss value for the epoch
    return total_loss / len(dl)

def eval_fn(m, dl, loss_fn, device):
    
    """
    
    This function gets model, dataloader, loss function, and device;
    conducts validation function and returns validation loss.
    
    Arguments:
    
        m        - model, torch model object;
        dl       - validation dataloader, torch dataloader object;
        loss_fn  - loss function, torch loss function object;
        device   - gpu device name, str.
        
    Outputs:
    
        loss     - loss value for the validation set in a specific epoch, float.
    
    """
    
    # Switch to evaluation mode
    m.eval()
    
    # Set initial loss value
    total_loss = 0.
    
    # Turn off gradient computation
    with torch.no_grad():
        
        # Go through validation dataloader        
        for q, p, n in tqdm(dl):
            
            # Move query, positive, and negative images to gpu
            q, p, n = q.to(device),p.to(device),n.to(device)
            
            # Get feature maps of the images
            q_fms, p_fms, n_fms = m(q), m(p), m(n)
            
            # Compute the loss value based on the feature maps
            loss = loss_fn(q_fms, p_fms, n_fms)
            
            # Add loss value of the mini-batch to the total loss
            total_loss += loss.item()
            
    # Return average loss for the epoch
    return total_loss / len(dl)

def get_fm_csv(m, data_dir, qry_im_names, device):
    
    """
    
    This function gets model, directory path, image names, and device;
    computes feature maps and returns dataframe with the calculated information.
    
    Arguments:
    
        m            - a trained model, torch model object;
        data_dir     - directory with the data;
        qry_im_names - names of the query images;
        device       - gpu device name.
        
    Outputs:
    
        df_enc       - feature maps, dataframe.

    """
    
    # Get images names as a numpy array
    qry_im_names_arr = np.array(qry_im_names)
    
    # Create a list for the feature maps
    fms = []
    
    # Switch the model into evaluation mode
    m.eval()
    
    # Turn off graduation computation
    with torch.no_grad():
        
        # Go through the image names array
        for i in tqdm(qry_im_names_arr):
            
            # Read query image
            qry = io.imread(data_dir + i)
            
            # Convert the images to tensor
            qry = (torch.from_numpy(qry).permute(2, 0, 1) / 255.).to(device)
            
            # Get feature maps 
            qry_fm = m(qry.unsqueeze(0))
            
            # Add obtained feature maps to the list
            fms.append(qry_fm.squeeze().cpu().detach().numpy())
        
        # Change the list to numpy array            
        fms = np.array(fms)
        
        # Create a dataframe 
        fms = pd.DataFrame(fms)
        
        # Concatenate the dataframe with the image names
        df_enc = pd.concat([qry_im_names, fms], axis=1)
        
    return df_enc

def euc_dist(fm, qry_fm_arr):
    
    """
    
    This function gets two feature maps and returns Euclidean distance between them.
    
    Arguments:
    
        fm          - feature map, tensor;
        qry_fm_arr  - query image feature map, array.
        
    Output:
    
        dist        - computed distance, float.
    
    """
    
    dist = np.sqrt(np.dot(fm - qry_fm_arr, (fm - qry_fm_arr).T))
    
    return dist

def plot_closest_imgs(qry_img_names, data_dir, image, img_path, closest_idx, distance, no_of_closest = 10):
    
    """
    
    This function gets query image names, directory path, image, image path, closest index, distance, and number of closest value
    and visualizes the images that are the closest to the query image.
    
    Arguments:
    
        qry_img_names   - names of the query images;
        data_dir        - path to directory with images;
        image           - an image;
        img_path        - an image path;
        closest_idx     - index of the closest feature map;
        distance        - computed distance;
        no_of_closest   - number of closest images to visualize.    
        
    Output:
    
        plot.
    
    """

    # Initialize grap
    G = nx.Graph()

    # Get name from the image path
    S_name = [img_path.split('/')[-1]]
    
    # Go through number of images to visualize
    for s in range(no_of_closest):
        S_name.append(qry_img_names.iloc[closest_idx[s]])

    # Go through the name
    for i in range(len(S_name)):
        # Read image
        image = io.imread(data_dir + S_name[i])
        # Add to the node
        G.add_node(i,image = image)
    
    # Add another node
    for j in range(1, no_of_closest + 1):
        G.add_edge(0, j , weight = distance[closest_idx[j - 1]])
    
    # Positive images
    pos = nx.kamada_kawai_layout(G)

    # Initialize figure
    fig = plt.figure(figsize = (20, 20))
    
    # Add subplots
    ax = plt.subplot(111)
    ax.set_aspect('equal')
    nx.draw_networkx_edges(G, pos, ax = ax)
    
    # Set limits to the coordinates
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)

    # Initialize transformations
    trans = ax.transData.transform
    trans2 = fig.transFigure.inverted().transform

    # Set image size
    piesize = 0.1 
    p2 = piesize / 2.0
    
    # Go through the node
    for n in G:
        
        # Get figure coordinates
        xx, yy = trans(pos[n]) 
        
        # Get axes coordinates
        xa,ya = trans2((xx, yy))
        a = plt.axes([xa - p2,ya - p2, piesize, piesize])
        a.set_aspect('equal')
        a.imshow(G.nodes[n]['image'])
        a.set_title(S_name[n][0:4])
        a.axis('off')
    
    # Turn off axis
    ax.axis('off')
    
    # Show the plot
    plt.show()
