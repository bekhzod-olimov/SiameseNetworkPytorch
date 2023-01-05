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

        # compute the loss
        loss = loss_fn(q_fms, p_fms, n_fms)
        
        opt.zero_grad()
        # backprop
        loss.backward()
        # optimization
        opt.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dl)


def eval_fn(m, dl, loss_fn, device):
    
    # switch to eval mode
    m.eval()
    
    total_loss = 0.
    
    with torch.no_grad():
        
        for q,p,n in tqdm(dl):
            
            q,p,n = q.to(device),p.to(device),n.to(device)
            
            # get fms
            q_fms = m(q)
            p_fms = m(p)
            n_fms = m(n)
            
            # compute the loss
            loss = loss_fn(q_fms, p_fms, n_fms)
            total_loss += loss.item()
            
    return total_loss / len(dl)


def get_fm_csv(m, data_dir, qry_im_names, device):
    
    # get im names as an array
    qry_im_names_arr = np.array(qry_im_names)
    fms = []
    
    m.eval()    
    with torch.no_grad():
        for i in tqdm(qry_im_names_arr):
            # read a qry image
            qry = io.imread(data_dir + i)
            # convert to tensor and move from cpu to gpu
            qry = (torch.from_numpy(qry).permute(2, 0, 1) / 255.).to(device)
            # get fms
            qry_fm = m(qry.unsqueeze(0))
            
            fms.append(qry_fm.squeeze().cpu().detach().numpy())
            
        fms = np.array(fms)
        # create a df
        fms = pd.DataFrame(fms)
        df_enc = pd.concat([qry_im_names, fms], axis=1)
        
    return df_enc

def euc_dist(fm, qry_fm_arr):
    
    dist = np.sqrt(np.dot(fm - qry_fm_arr, (fm - qry_fm_arr).T))
    
    return dist


def plot_closest_imgs(anc_img_names, DATA_DIR, image, img_path, closest_idx, distance, no_of_closest = 10):

    G=nx.Graph()

    S_name = [img_path.split('/')[-1]]

    for s in range(no_of_closest):
        S_name.append(anc_img_names.iloc[closest_idx[s]])

    for i in range(len(S_name)):
        image = io.imread(DATA_DIR + S_name[i])
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
