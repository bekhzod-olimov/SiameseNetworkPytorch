import torch, timm

class Model(torch.nn.Module):
    
    """
    
    This class gets a model name and feature embedding size and returns a model to be trained.
    
    Arguments:
    
        model_name   - a name of the model in timm librariy to be trained, str;
        emb_size     - feature embedding dimension, int.
        
    Output:
    
        fms          - feature map, tensor.
    
    """
    # Make changes here
    # Initialize model with model name and embedding size
    def __init__(self, model_name, emb_size = 512):
        super(Model, self).__init__()
        
        # Get the base model
        self.eff = timm.create_model(model_name, pretrained=True)
        
        # Change the classifier part of the base model
        self.eff.classifier = torch.nn.Linear(in_features=self.eff.classifier.in_features, out_features=emb_size)
        print(f"\nModel {model_name} is successfully loaded!")
        
    # Get feature maps and return them as an output
    
    def forward(self, x): return self.eff(x)
