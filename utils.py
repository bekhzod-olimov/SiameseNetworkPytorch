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
    
    Gets model, dataloader, optimizer, loss function, and gpu device;
    conducts one epoch of training;
    and return train loss for the epoch.
    
    Arguments:
    m - model;
    dl - train dataloader;
    opt - optimizer;
    loss_fn - loss function;
    device - gpu device.
    
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


# Validation function
def eval_fn(m, dl, loss_fn, device):
    
    """
    
    Gets model, dataloader, loss function, and device;
    conducts validation function and returns validation loss.
    
    Arguments:
    m - model;
    dl - validation dataloader;
    loss_fn - loss function;
    device - gpu device name.
    
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
            q_fms = m(q)
            p_fms = m(p)
            n_fms = m(n)
            
            # Compute the loss value based on the feature maps
            loss = loss_fn(q_fms, p_fms, n_fms)
            
            # Add loss value of the mini-batch to the total loss
            total_loss += loss.item()
            
    # Return average loss for the epoch
    return total_loss / len(dl)

# Function to create csv file
def get_fm_csv(m, data_dir, qry_im_names, device):
    
    """
    
    Gets model, directory path, image names, and device;
    computes feature maps and returns dataframe with the 
    calculated information.
    
    Arguments:
    m - trained model;
    data_dir - directory with the data;
    qry_im_names - names of the query images;
    device - gpu device name.
    
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

# Euclidean distance function
def euc_dist(fm, qry_fm_arr):
    
    """
    
    Gets two feature maps and returns Euclidean distance between them.
    
    Arguments:
    fm - feature map;
    qry_fm_arr - query image feature map
    
    """
    
    dist = np.sqrt(np.dot(fm - qry_fm_arr, (fm - qry_fm_arr).T))
    
    return dist


# Visualization function
def plot_closest_imgs(qry_img_names, data_dir, image, img_path, closest_idx, distance, no_of_closest = 10):
    
    """
    
    Gets query image names, directory path, image, image path, closest index, distance, and number of closest value
    and visualizes the images that are the closest to the query image.
    
    Arguments:
    qry_img_names - names of the query images;
    data_dir - path to directory with images;
    image - image;
    img_path - image path;
    closest_idx - index of the closest feature map;
    distance - distance;
    no_of_closest - number of closest images to visualize.    
    
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
        image = io.imread(data_dir + S_name[i])
        G.add_node(i,image = image)
        
    for j in range(1,no_of_closest + 1):
        G.add_edge(0,j,weight=distance[closest_idx[j-1]])
        

    pos=nx.kamada_kawai_layout(G)

    fig=plt.figure(figsize=(20,20))
    ax=plt.subplot(111)
    ax.set_aspect('equal')
    nx.draw_networkx_edges(G,pos,ax=ax)

    plt.xlim(-1.5,1.5)
    plt.ylim(-1.5,1.5)

    trans=ax.transData.transform
    trans2=fig.transFigure.inverted().transform

    piesize=0.1 # this is the image size
    p2=piesize/2.0
    for n in G:
        xx,yy=trans(pos[n]) # figure coordinates
        xa,ya=trans2((xx,yy)) # axes coordinates
        a = plt.axes([xa-p2,ya-p2, piesize, piesize])
        a.set_aspect('equal')
        a.imshow(G.nodes[n]['image'])
        a.set_title(S_name[n][0:4])
        a.axis('off')
    ax.axis('off')
    plt.show()
