# Import libraries
import cv2, argparse, torch, sys, yaml, os
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split 
from torch.utils.data import DataLoader; from dataset import CustomDataset
from model import Model; from utils import train_fn, eval_fn

def run(args):
    
    # Get train arguments
    backbone = args.backbone
    epochs = args.epochs
    device = args.device
    path = args.ims_path
    bs = args.batch_size
    lr = args.learning_rate    
    save_path = args.save_path
    root = args.root
    
    argstr = yaml.dump(args.__dict__, default_flow_style = False)
    print(f"\nTraining Arguments:\n{argstr}\n")
    
    # Read the data
    df = pd.read_csv(path)
    
    # Split the data
    train_df, valid_df = train_test_split(df, test_size = 0.2, random_state = 2023)
    
    # Get train and validation sets
    tr_ds, val_ds = CustomDataset(root, train_df), CustomDataset(root, valid_df)
    print(f"Number of training samples: {len(tr_ds)}")
    print(f"Number of validation samples: {len(val_ds)}")
    
    # Create train and validation dataloaders
    trainloader, validloader = DataLoader(tr_ds, batch_size = bs, shuffle = True), DataLoader(val_ds, batch_size = bs, shuffle = False)
    
    # Get a training model
    model = Model(backbone)
    model.to(device)
    
    # Loss function 
    loss_fn = torch.nn.TripletMarginLoss()
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)    
    
    # Training loop
    best_loss = np.Inf    
    for epoch in range(epochs):
        
        # Get train and validation losses
        train_loss = train_fn(model, trainloader, optimizer, loss_fn, device)
        valid_loss = eval_fn(model, validloader, loss_fn, device)
        
        # Save a model with the lowest loss value
        if valid_loss < best_loss:
            torch.save(model.state_dict(), 'best_model.pt')
            best_loss = valid_loss
            print("The best model is saved!")
    
    # Verbose           
    print(f"{epoch + 1} is completed with the train loss of {train_loss:.3f} and the validation loss of {valid_loss:.3f}")
    
if __name__ == "__main__":
    
    # Initialize Argument Parser Object
    parser = argparse.ArgumentParser(description='Siamese Network Training Arguments')
    
    # Add arguments to the Parser
    parser.add_argument("-sp", "--save_path", type=str, default='saved_models', help="Path to save trained models")
    parser.add_argument("-bs", "--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("-d", "--device", type=str, default='cuda:2', help="GPU device number")
    parser.add_argument("-ip", "--ims_path", type=str, default='data/train.csv', help="Path to the images")
    parser.add_argument("-r", "--root", type=str, default='data/train/', help="Path to the data")
    parser.add_argument("-bb", "--backbone", type=str, default='efficientnet_b0', help="Model name for backbone")
    parser.add_argument("-w", "--weights", type=str, default='imagenet', help="Pretrained weights type")
    parser.add_argument("-lr", "--learning_rate", type=float, default=3e-3, help="Learning rate value")
    parser.add_argument("-e", "--epochs", type=int, default=200, help="Number of epochs")
    
    # Parse the added arguments
    args = parser.parse_args() 
    
    # Run the script with the arguments
    run(args)
