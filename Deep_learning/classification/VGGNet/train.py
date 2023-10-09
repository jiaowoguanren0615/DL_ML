from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
from torch.utils.data import DataLoader
from config import device, epochs, root, batch_size, lr, weight_decay, save_path, data_transform, resume, best_val_accuracy
from utils import read_split_data, train_step, val_step, Plot_ROC
from my_dataset import MyDataset
from model import make_model
from VGGPredict import predictor, predict_single_image


train_image_path, train_image_label, val_image_path, val_image_label, class_indices = read_split_data(root)

train_dataset = MyDataset(train_image_path, train_image_label, data_transform['train'])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                          collate_fn=train_dataset.collate_fn)

valid_dataset = MyDataset(val_image_path, val_image_label, data_transform['valid'])
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
                          collate_fn=valid_dataset.collate_fn)


net = make_model(num_classes=5).to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

"""
def scheduler(now_epoch):
    end_lr_rate = 0.01
    rate = ((1 + math.cos(now_epoch * math.pi / epochs)) / 2) * (1 - end_lr_rate) + end_lr_rate
    new_lr = rate * initial_lr
    return new_lr
"""

lr_scheduler = CosineAnnealingLR(optimizer, T_max=5)

if resume:
    checkpoint = torch.load(save_path)
    best_val_accuracy = checkpoint['best_accuracy']
    net.load_state_dict(checkpoint['model'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

scalar = torch.cuda.amp.GradScaler() if torch.cuda.is_bf16_supported() else None

for epoch in range(epochs):
    # train
    train_loss, train_accuracy = train_step(net, optimizer, train_loader, device, epoch, scalar)
    # valid
    val_loss, val_accuracy = val_step(net, valid_loader, device, epoch)

    lr_scheduler.step()

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        save_parameters = {
            'model': net.state_dict(),
            'best_accuracy': val_accuracy,
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict()
        }

        torch.save(save_parameters, save_path)

print('Now we predict an image!!!')
predict_single_image()
print('\n')
f1score = predictor(valid_loader)
Plot_ROC(net, valid_loader, save_path, device)