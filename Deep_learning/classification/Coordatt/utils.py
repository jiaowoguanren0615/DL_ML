import sys, json
import torch
from tqdm import tqdm
import torch.nn as nn
from torchinfo import summary
from sklearn.metrics import auc, roc_curve
import numpy as np
import matplotlib.pyplot as plt
from numpy import interp
from itertools import cycle


def train_step(net, optimizer, data_loader, device, epoch, scalar=None):
    net.train()
    loss_function = nn.CrossEntropyLoss()
    train_acc, train_loss, sampleNum = 0, 0, 0
    optimizer.zero_grad()

    train_bar = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(train_bar):
        images, labels = data
        sampleNum += images.shape[0]  # batch
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if scalar is not None:
            with torch.cuda.amp.autocast():
                outputs = net(images)
                loss = loss_function(outputs, labels)
        else:
            outputs = net(images)
            loss = loss_function(outputs, labels)

        train_acc += (torch.argmax(outputs, dim=1) == labels).sum().item()
        train_loss += loss.item()
        # loss.backward()
        # optimizer.step()

        if scalar is not None:
            scalar.scale(loss).backward()
            scalar.step(optimizer)
            scalar.update()
        else:
            loss.backward()
            optimizer.step()
        train_bar.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch, train_loss / (step + 1),
                                                                             train_acc / sampleNum)

    return train_loss / (step + 1), train_acc / sampleNum


@torch.no_grad()
def val_step(net, data_loader, device, epoch):
    loss_function = nn.CrossEntropyLoss()
    net.eval()
    val_acc = 0
    val_loss = 0
    sample_num = 0
    val_bar = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(val_bar):
        images, labels = data
        sample_num += images.shape[0]
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        loss = loss_function(outputs, labels)
        val_loss += loss.item()
        val_acc += (torch.argmax(outputs, dim=1) == labels).sum().item()
        val_bar.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch, val_loss / (step + 1),
                                                                           val_acc / sample_num)

    return val_loss / (step + 1), val_acc / sample_num


def get_net_summary(net, input_size):
    return summary(net, input_size, col_names=["input_size",
                                               "output_size",
                                               "num_params",
                                               "params_percent",
                                               "kernel_size",
                                               "mult_adds",
                                               "trainable"])


def Plot_ROC(net, val_loader, save_name, device):
    try:
        json_file = open('./classes_indices.json', 'r')
        class_indict = json.load(json_file)
    except Exception as e:
        print(e)
        exit(-1)

    score_list = []  # 存储预测得分
    label_list = []  # 存储真实标签

    net.load_state_dict(torch.load(save_name)['model'])

    for i, data in enumerate(val_loader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = torch.softmax(net(images), dim=1)
        score_tmp = outputs
        score_list.extend(score_tmp.detach().cpu().numpy())
        label_list.extend(labels.cpu().numpy())

    score_array = np.array(score_list)
    # 将label转换成onehot形式
    label_tensor = torch.tensor(label_list)
    label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
    label_onehot = torch.zeros(label_tensor.shape[0], len(class_indict.keys()))
    label_onehot.scatter_(dim=1, index=label_tensor, value=1)
    label_onehot = np.array(label_onehot)

    print("score_array:", score_array.shape)  # (batchsize, classnum)
    print("label_onehot:", label_onehot.shape)  # torch.Size([batchsize, classnum])

    # 调用sklearn库，计算每个类别对应的fpr和tpr
    fpr_dict = dict()
    tpr_dict = dict()
    roc_auc_dict = dict()
    for i in range(len(class_indict.keys())):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(label_onehot[:, i], score_array[:, i])
        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
    # micro
    fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve(label_onehot.ravel(), score_array.ravel())
    roc_auc_dict["micro"] = auc(fpr_dict["micro"], tpr_dict["micro"])

    # macro
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(len(class_indict.keys()))]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(len(set(label_list))):
        mean_tpr += interp(all_fpr, fpr_dict[i], tpr_dict[i])

    # Finally average it and compute AUC
    mean_tpr /= len(class_indict.keys())
    fpr_dict["macro"] = all_fpr
    tpr_dict["macro"] = mean_tpr
    roc_auc_dict["macro"] = auc(fpr_dict["macro"], tpr_dict["macro"])

    # 绘制所有类别平均的roc曲线
    plt.figure(figsize=(12, 12))
    lw = 2

    plt.plot(fpr_dict["micro"], tpr_dict["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc_dict["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr_dict["macro"], tpr_dict["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc_dict["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(len(class_indict.keys())), colors):
        plt.plot(fpr_dict[i], tpr_dict[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(class_indict[str(i)], roc_auc_dict[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw, label='Chance', color='red')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig('./save_images/multi_classes_roc.png')
    # plt.show()
