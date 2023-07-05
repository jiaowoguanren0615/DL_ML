import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import data_transform, epochs, save_path, batch_size, lr, weight_decay, root, devices, num_gpus
from ..ViT.my_dataset import MyDataset
from model import vit_base_patch16_224 as make_model
from ..ViT.split_data import read_split_data
import sys


def train(net, batch_size, num_gpus, lr, weight_decay, epochs, save_path):
    assert num_gpus > 1, 'You must have 2 or more gpus in your machine !'

    best_val_acc = 0

    net = nn.DataParallel(net, device_ids=devices).to(devices[0])

    train_image_path, train_image_label, val_image_path, val_image_label, class_indices = read_split_data(root)
    train_dataset = MyDataset(train_image_path, train_image_label, data_transform['train'])
    train_num = len(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                              collate_fn=train_dataset.collate_fn)

    valid_dataset = MyDataset(val_image_path, val_image_label, data_transform['valid'])
    valid_num = len(valid_dataset)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                              collate_fn=valid_dataset.collate_fn)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        running_loss, train_acc, val_acc = 0, 0, 0

        net.train()
        train_bar = tqdm(train_loader, file=sys.stdout, colour='red')
        for step, data in enumerate(train_bar):
            images, labels = data
            images, labels = images.to(devices[0]), labels.to(devices[0])
            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_function(outputs, labels)
            running_loss += loss.item()
            train_acc += (torch.argmax(outputs, dim=1) == labels).sum().item()
            loss.backward()
            optimizer.step()
            train_bar.desc = f'train epoch [{epoch+1}/{epochs}] loss: {loss.item():.3f}'

        net.eval()
        with torch.no_grad():
            val_bar = tqdm(valid_loader, file=sys.stdout, colour='red')
            for data in val_bar:
                images, labels = data
                images, labels = images.to(devices[0]), labels.to(devices[0])
                outputs = net(images)
                val_acc += (torch.argmax(outputs, dim=1) == labels).sum().item()

        train_accuracy = train_acc / train_num
        val_accuracy = val_acc / valid_num
        train_loss = running_loss / train_num

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(net.state_dict(), save_path)

        print(f'epoch [{epoch+1}/{epochs}], train_loss: {train_loss:.3f}, train_accuracy: {train_accuracy:.3f}, valid_accuracy: {val_accuracy:.3f}')

    print('Finished Training!!!')


if __name__ == '__main__':
    net = make_model(num_classes=5)

    # if os.path.exists(save_path):
    #     print('----------------------------LoadingModel----------------------------')
    #     net.load_state_dict(torch.load(save_path), strict=False)

    train(net=net, batch_size=batch_size, num_gpus=num_gpus,
          lr=lr, weight_decay=weight_decay, epochs=epochs, save_path=save_path)