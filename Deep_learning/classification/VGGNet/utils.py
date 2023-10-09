import os, cv2, json, random, sys
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import auc, roc_curve
import numpy as np
from numpy import interp
from itertools import cycle



def read_split_data(root, plot_image=False):
    filepaths = []
    labels = []
    bad_images = []

    random.seed(0)
    assert os.path.exists(root), 'Your root does not exists!!!'

    classes = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    classes.sort()
    class_indices = {k: v for v, k in enumerate(classes)}

    json_str = json.dumps({v: k for k, v in class_indices.items()}, indent=4)

    with open('./classes_indices.json', 'w') as json_file:
        json_file.write(json_str)

    every_class_num = []
    supported = ['.jpg', '.png', '.jpeg', '.PNG', '.JPG', '.JPEG']

    for klass in classes:
        classpath = os.path.join(root, klass)
        images = [os.path.join(root, klass, i) for i in os.listdir(classpath) if os.path.splitext(i)[-1] in supported]
        every_class_num.append(len(images))
        flist = sorted(os.listdir(classpath))
        desc = f'{klass:23s}'
        for f in tqdm(flist, ncols=110, desc=desc, unit='file', colour='blue'):
            fpath = os.path.join(classpath, f)
            fl = f.lower()
            index = fl.rfind('.')
            ext = fl[index:]
            if ext in supported:
                try:
                    img = cv2.imread(fpath)
                    filepaths.append(fpath)
                    labels.append(klass)
                except:
                    bad_images.append(fpath)
                    print('defective image file: ', fpath)
            else:
                bad_images.append(fpath)

    Fseries = pd.Series(filepaths, name='filepaths')
    Lseries = pd.Series(labels, name='labels')
    df = pd.concat([Fseries, Lseries], axis=1)

    print(f'{len(df.labels.unique())} kind of images were found in the dataset')
    train_df, test_df = train_test_split(df, train_size=.8, shuffle=True, random_state=123, stratify=df['labels'])

    train_image_path = train_df['filepaths'].tolist()
    val_image_path = test_df['filepaths'].tolist()

    train_image_label = [class_indices[i] for i in train_df['labels'].tolist()]
    val_image_label = [class_indices[i] for i in test_df['labels'].tolist()]

    sample_df = train_df.sample(n=50, replace=False)
    ht, wt, count = 0, 0, 0
    for i in range(len(sample_df)):
        fpath = sample_df['filepaths'].iloc[i]
        try:
            img = cv2.imread(fpath)
            h = img.shape[0]
            w = img.shape[1]
            ht += h
            wt += w
            count += 1
        except:
            pass
    have = int(ht / count)
    wave = int(wt / count)
    aspect_ratio = have / wave
    print('{} images were found in the dataset.\n{} for training, {} for validation'.format(
        sum(every_class_num), len(train_image_path), len(val_image_path)
    ))
    print('average image height= ', have, '  average image width= ', wave, ' aspect ratio h/w= ', aspect_ratio)

    if plot_image:
        plt.bar(range(len(classes)), every_class_num, align='center')
        plt.xticks(range(len(classes)), classes)

        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')

        plt.xlabel('image class')
        plt.ylabel('number of images')

        plt.title('class distribution')
        plt.show()

    return train_image_path, train_image_label, val_image_path, val_image_label, class_indices



def train_step(net, optimizer, data_loader, device, epoch, scalar=None):

    net.train()
    loss_function = nn.CrossEntropyLoss()
    train_acc = 0
    train_loss = 0
    sample_num = 0

    data_loader = tqdm(data_loader, file=sys.stdout, colour='red')
    for step, data in enumerate(data_loader):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        sample_num += images.shape[0]

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

        if scalar is not None:
            scalar.scale(loss).backward()
            scalar.step(optimizer)
            scalar.update()
        else:
            loss.backward()
            optimizer.step()
        data_loader.desc = "[train epoch {}] loss: {:.3f}, accuracy: {:.3f}".format(epoch+1,
                                                                               train_loss / (step + 1),
                                                                               train_acc / sample_num)

    return train_loss / (step + 1), train_acc / sample_num



@torch.no_grad()
def val_step(net, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    net.eval()
    val_acc = 0
    val_loss = 0
    sample_num = 0

    data_loader = tqdm(data_loader, file=sys.stdout, colour='red')
    for step, data in enumerate(data_loader):
        images, labels = data

        images = images.to(device)
        labels = labels.to(device)
        sample_num += images.shape[0]

        outputs = net(images)
        loss = loss_function(outputs, labels)
        val_loss += loss.item()
        val_acc += (torch.argmax(outputs, dim=1) == labels).sum().item()

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, accuracy: {:.3f}".format(epoch+1,
                                                                               val_loss / (step + 1),
                                                                               val_acc / sample_num)

    return val_loss / (step + 1), val_acc / sample_num


@torch.no_grad()
def Plot_ROC(net, val_loader, save_name, device):
    try:
        json_file = open('./classes_indices.json', 'r')
        class_indict = json.load(json_file)
    except Exception as e:
        print(e)
        exit(-1)

    score_list = []
    label_list = []

    net.load_state_dict(torch.load(save_name)['model'])
    for i, data in enumerate(val_loader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = torch.softmax(net(images), dim=1)
        score_tmp = outputs
        score_list.extend(score_tmp.detach().cpu().numpy())
        label_list.extend(labels.cpu().numpy())

    score_array = np.array(score_list)

    label_tensor = torch.tensor(label_list)
    label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
    label_onehot = torch.zeros(label_tensor.shape[0], len(class_indict.keys()))
    label_onehot.scatter_(dim=1, index=label_tensor, value=1)
    label_onehot = np.array(label_onehot)

    print("score_array:", score_array.shape)  # (batchsize, classnum)
    print("label_onehot:", label_onehot.shape)  # torch.Size([batchsize, classnum])

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

    # plot roc curve, contains all classes
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
    plt.savefig('./multi_classes_roc.png')