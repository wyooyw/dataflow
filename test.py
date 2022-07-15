import torch
if __name__=="__main__":
    tensor = torch.ones((4,4))
    print(tensor)
    tensor[0] = torch.zeros(5)
    print(tensor)