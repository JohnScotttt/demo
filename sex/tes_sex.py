import torch
import torch.nn as nn

class sexnet(nn.Module):
    def __init__(self):
        super(sexnet, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(3, 3),
            nn.ReLU(),
            nn.Linear(3, 2)
        )

    def forward(self, x):
        out = self.dense(x)
        return out


dic1={0:'woman',1:'man'}
model=sexnet()
model.load_state_dict(torch.load('output/params_100.pth'))
model.eval()

height=float(input('height: '))/2.0
weight=float(input('weight: '))/80.0
waist=float(input("waist:"))/90.0
ttenser=torch.tensor([height,weight,waist])
out=model(ttenser)
pred=torch.max(out,0)[1]
print(dic1[pred.item()])