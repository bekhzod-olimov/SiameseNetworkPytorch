# Import libraries
import torch
from skimage import io
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    
    """
    
    This class gets a directory path with images and dataframe and returns dataset.
    
    Parameters:
    
        data_dir - a path to the data;
        df       - dataframe with data information.
        
    Output:
    
        dataset, torch dataset object.
    
    """
    
    # Initialize with path to the data, and dataframe with the dataset information
    def __init__(self, data_dir, df): 
        
        self.data_dir, self.df = data_dir, df
        
    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):

        # Get dataset row based on the index        
        example = self.df.iloc[idx]
        
        # Read query, positive, and negative images
        qry_img, pos_img, neg_img = io.imread(self.data_dir + example.Anchor), io.imread(self.data_dir + example.Positive), io.imread(self.data_dir + example.Negative)
        
        # Transform them into tensors and return
        return torch.from_numpy(qry_img).permute(2,0,1) / 255., torch.from_numpy(pos_img).permute(2,0,1) / 255., torch.from_numpy(neg_img).permute(2,0,1) / 255.
