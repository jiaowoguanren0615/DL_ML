import sys, torch
from tqdm import tqdm
import torch
import torch.nn as nn
from .distributed_utils import reduce_value, is_main_process


def train_step(net, optimizer, data_loader, device, epoch, scalar=None):
    net.train()
    loss_function = nn.CrossEntropyLoss()
    sampleNum, mean_loss = 0, torch.zeros(1).to(device)
    optimizer.zero_grad()

    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data
        sampleNum += images.shape[0]  # batch
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if scalar is not None:
            with torch.cuda.amp.autocast():
                outputs = net(images)
                loss = loss_function(outputs, labels)
                loss = reduce_value(loss, average=True)
        else:
            outputs = net(images)
            loss = loss_function(outputs, labels)
            loss = reduce_value(loss, average=True)

        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)

        # if not torch.isfinite(loss):
        #     print('WARNING: non-finite loss, ending training ', loss)
        #     sys.exit(1)

        if is_main_process():
            data_loader.desc = "[epoch {}] mean loss: {}".format(epoch, round(mean_loss.item(), 3))

        if scalar is not None:
            scalar.scale(loss).backward()
            scalar.step(optimizer)
            scalar.update()
        else:
            loss.backward()
            optimizer.step()

    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return mean_loss.item()


@torch.no_grad()
def val_step(net, data_loader, device):
    net.eval()
    val_acc = torch.zeros(1).to(device)

    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)

        val_acc += (torch.argmax(outputs, dim=1) == labels).sum()

    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    val_acc = reduce_value(val_acc, average=False)

    return val_acc.item()
