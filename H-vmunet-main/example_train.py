import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.light_medseg_net import LightSegNet, reparameterize_replk
from losses.cldice import SoftDiceLoss, clDiceLoss, cbDiceLoss

# toy random dataset for demonstration
class ToySet(torch.utils.data.Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, idx):
        img = torch.rand(3, 256, 256)
        mask = torch.rand(1, 64, 64)
        return img, mask

train_loader = DataLoader(ToySet(), batch_size=2)

model = LightSegNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

bce = nn.BCELoss()
dice = SoftDiceLoss()
cldice = clDiceLoss()
cbdice = cbDiceLoss()

for epoch in range(1):
    for img, mask in train_loader:
        seg, edge, thick = model(img)
        loss = (
            bce(seg, mask) +
            dice(seg, mask) +
            cldice(seg, mask) +
            cbdice(seg, mask) +
            F.mse_loss(thick.squeeze(), torch.zeros_like(thick.squeeze()))
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('loss:', loss.item())

# reparameterize after training if needed
# reparameterize_replk(model)

# export onnx
# import onnx
