import torch, timm

class Model(torch.nn.Module):
    
    # Initialize model with model name and embedding size
    def __init__(self, model_name, emb_size=512):
        super(Model, self).__init__()
        
        # Get the base model
        self.eff = timm.create_model(model_name, pretrained=True)
        
        # Change the classifier part of the base model
        self.eff.classifier = torch.nn.Linear(in_features=self.eff.classifier.in_features, out_features=emb_size)
        print(f"\nModel {model_name} is successfully loaded!")
        
    def forward(self, x):
        
        # Get feature maps and return them as an output
        fms = self.eff(x)
        
        return fms

