import torch
import numpy as np
from torch.utils.data import DataLoader,TensorDataset
from lossFunc import mape_loss
from readData import getDataforTest
net = torch.load('MultiDet.pt')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_x_data, _y_data = getDataforTest()
#########################################################

batch_size = 72
test_dataset = TensorDataset(_x_data, _y_data)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


test_loaderlen = len(test_loader)
for i, data in enumerate(test_loader):

    inputs, labels = data
    inputs, labels = inputs.to(device).float(), labels.to(device).float()

    label1 = torch.flatten(input=labels[:, :, : 1], start_dim=1)
    label2 = torch.flatten(input=labels[:, :, 1: 2], start_dim=1)
    label3 = torch.flatten(input=labels[:, :, 2: 3], start_dim=1)

    output1, output2, output3 = net(src=inputs)

loss1 = mape_loss(label1, output1)
loss2 = mape_loss(label2, output2)
loss3 = mape_loss(label3, output3)

print(label1.shape)
label1=label1.reshape(72*24,1)
np.savetxt('label1.csv', label1,fmt='%.2f',delimiter=',')
# print(output1.shape)
output1=output1.detach().numpy()
output1=output1.reshape(72*24,1)
np.savetxt('output1.csv', output1,fmt='%.2f',delimiter=',')
