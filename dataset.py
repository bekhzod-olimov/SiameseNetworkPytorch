import torch
from skimage import io
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    
    # Initialize with path to the data, and dataframe with the dataset information
    def __init__(self, data_dir, df): 
        
        self.data_dir = data_dir
        self.df = df
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        
        # Get dataset row based on the index        
        example = self.df.iloc[idx]
        
        # Read query, positive, and negative images
        qry_img = io.imread(self.data_dir + example.Anchor)
        pos_img = io.imread(self.data_dir + example.Positive)
        neg_img = io.imread(self.data_dir + example.Negative)
        
        # Transform them into tensors
        qry_img = torch.from_numpy(qry_img).permute(2,0,1) / 255.
        pos_img = torch.from_numpy(pos_img).permute(2,0,1) / 255.
        neg_img = torch.from_numpy(neg_img).permute(2,0,1) / 255.

        return qry_img, pos_img, neg_img
