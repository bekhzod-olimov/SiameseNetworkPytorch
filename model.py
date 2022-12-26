import torch, timm

class Model(torch.nn.Module):
    
    def __init__(self, model_name, emb_size=512):
        super(Model, self).__init__()
        self.eff = timm.create_model(model_name, pretrained=True)
        self.eff.classifier = torch.nn.Linear(in_features=self.eff.classifier.in_features, out_features=emb_size)
        print(f"\nModel {model_name} is successfully loaded!")
        
    def forward(self, inp):
        
        fms = self.eff(inp)
        
        return fms

