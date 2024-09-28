import torch 
try:
    checkpoint = torch.load('global_model.pth', map_location=torch.device('cpu'))
    print(checkpoint.keys())
except Exception as e:
    print(e)
