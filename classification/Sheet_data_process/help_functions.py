import torch
import sys
import os, json
from numpy import interp
import warnings
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchinfo import summary
from sklearn.metrics import auc, f1_score, roc_curve, classification_report, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from itertools import cycle
import pandas as pd

warnings.filterwarnings('ignore')


class NeuralNetwork():
    def __init__(self,
                 data_path=r'./breast-cancer.csv',
                 epochs=20, predictor=True, save_path='Model.pth', best_val_acc=0,
                 dfAccuracy_save_path='./dfAccuracy.csv', plot_roc=True):

        """
        :param data_path: Your csv or xlsx data_path
        :param epochs: your train epochs
        :param predictor: Do you need predict your test_data and plot confusion_matrix (default True)
        :param save_path: Your model save_name
        :param best_val_acc: initial a best_val_accuracy for comparing val_accuracy every epoch (default 0)
        :param dfAccuracy_save_path: Create a csv file to write accuracy loss learning_rate every epoch
        :param plot_roc: Do you want to plot ROC curve (default True)
        """

        self.epochs = epochs
        self.data_path = data_path
        self.plot_roc = plot_roc
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tb_writer = SummaryWriter()
        self.lr = 0.01
        self.predictor = predictor
        self.save_path = save_path
        self.best_val_acc = best_val_acc

        self.dfAccuracy_save_path = dfAccuracy_save_path
        dfMerge = pd.read_csv(self.data_path)

        # {'0': 'B', '1': 'M'}
        self.target_dict = {k: v for k, v in enumerate(dfMerge['diagnosis'].unique())}

        class_indices = {k: v for v, k in enumerate(dfMerge['diagnosis'].unique())}
        json_str = json.dumps({v: k for k, v in class_indices.items()}, indent=4)
        with open('./classes_indices.json', 'w') as json_file:
            json_file.write(json_str)

        dfMerge['diagnosis'] = dfMerge['diagnosis'].map(class_indices)
        self.X = np.array(dfMerge.iloc[:, 2:])
        self.y = np.array(dfMerge['diagnosis'])

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y)
        self.X_train = torch.FloatTensor(X_train)
        self.X_test = torch.FloatTensor(X_test)
        self.y_train = torch.LongTensor(y_train)
        self.y_test = torch.LongTensor(y_test)


    def fit(self):
        class MyDataset(Dataset):
            def __init__(self, x_data, y_data):
                self.x_data = x_data
                self.y_data = y_data

            def __len__(self):
                return len(self.x_data)

            def __getitem__(self, item):
                X_sample = self.x_data[item]
                y_sample = self.y_data[item]
                return X_sample, y_sample

            @staticmethod
            def collate_fn(batch):
                x, y = tuple(zip(*batch))
                x = torch.stack(x, dim=0)
                #     x = torch.FloatTensor(x)
                y = torch.as_tensor(y)
                return x, y

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.features = nn.Sequential(
                    nn.Linear(30, 32),
                    nn.BatchNorm1d(32),
                    nn.ReLU(inplace=True),
                    nn.Linear(32, 2)
                )

            def forward(self, x):
                y = self.features(x)
                return y

        net = Net()
        # summary(net, input_size=(569, 30), col_names=["input_size", "output_size", "num_params",
        #         "kernel_size", "mult_adds", "trainable"])

        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        train_dataset = MyDataset(self.X_train, self.y_train)
        val_dataset = MyDataset(self.X_test, self.y_test)
        train_loader = DataLoader(
            train_dataset, batch_size=8, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=8,
                                shuffle=False, num_workers=0)

        tags = ['learning_rate', 'train_accuracy',
                'train_loss', 'val_accuracy', 'val_loss']

        scheduler = CosineAnnealingLR(optimizer, T_max=5)

        if os.path.exists(self.dfAccuracy_save_path):
            self.best_val_acc = max(pd.read_csv(self.dfAccuracy_save_path)['val_accuracy'])
            dfAccuracy = pd.DataFrame(index=list(
                range(self.epochs)), columns=tags)
        else:
            dfAccuracy = pd.DataFrame(index=list(range(self.epochs)), columns=tags)

        for epoch in range(self.epochs):
            train_loss, train_accuracy = self.train_step(net, train_loader, optimizer, epoch, self.device)
            val_loss, val_accuracy = self.val_step(net, val_loader, epoch, self.device)
            scheduler.step()

            self.tb_writer.add_scalar(tags[0], optimizer.param_groups[0]['lr'], epoch)
            self.tb_writer.add_scalar(tags[1], train_accuracy, epoch)
            self.tb_writer.add_scalar(tags[2], train_loss, epoch)
            self.tb_writer.add_scalar(tags[3], val_accuracy, epoch)
            self.tb_writer.add_scalar(tags[4], val_loss, epoch)

            dfAccuracy.loc[epoch, tags[0]] = optimizer.param_groups[0]['lr']
            dfAccuracy.loc[epoch, tags[1]] = train_accuracy
            dfAccuracy.loc[epoch, tags[2]] = train_loss
            dfAccuracy.loc[epoch, tags[3]] = val_accuracy
            dfAccuracy.loc[epoch, tags[4]] = val_loss

            if val_accuracy > self.best_val_acc:
                self.best_val_acc = val_accuracy
                torch.save(net.state_dict(), self.save_path)

        dfAccuracy.to_csv(self.dfAccuracy_save_path, index=False)
        if self.predictor:
            self.Predictor(net, val_loader)

        self.plot_acc_loss(dfAccuracy)

        if self.plot_roc:
            self.Plot_ROC(net, val_loader)


    def Plot_ROC(self, net, val_loader):
        try:
            json_file = open('./classes_indices.json', 'r')
            class_indict = json.load(json_file)
        except Exception as e:
            print(e)
            exit(-1)

        score_list = []  # 存储预测得分
        label_list = []  # 存储真实标签
        net.load_state_dict(torch.load(self.save_path))

        for i, data in enumerate(val_loader):
            x, y = data
            x, y = x.to(self.device), y.to(self.device)
            outputs = torch.softmax(net(x), dim=1)
            score_tmp = outputs
            score_list.extend(score_tmp.detach().cpu().numpy())
            label_list.extend(y.cpu().numpy())

        score_array = np.array(score_list)
        # 将label转换成onehot形式
        label_tensor = torch.tensor(label_list)
        label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
        label_onehot = torch.zeros(label_tensor.shape[0], len(set(label_list)))
        label_onehot.scatter_(dim=1, index=label_tensor, value=1)
        label_onehot = np.array(label_onehot)

        print("score_array:", score_array.shape)  # (batchsize, classnum)
        print("label_onehot:", label_onehot.shape)  # torch.Size([batchsize, classnum])

        # 调用sklearn库，计算每个类别对应的fpr和tpr
        fpr_dict = dict()
        tpr_dict = dict()
        roc_auc_dict = dict()
        for i in range(len(set(label_list))):
            fpr_dict[i], tpr_dict[i], _ = roc_curve(label_onehot[:, i], score_array[:, i])
            roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
        # micro
        fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve(label_onehot.ravel(), score_array.ravel())
        roc_auc_dict["micro"] = auc(fpr_dict["micro"], tpr_dict["micro"])

        # macro
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(len(set(label_list)))]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)

        for i in range(len(set(label_list))):
            mean_tpr += interp(all_fpr, fpr_dict[i], tpr_dict[i])

        # Finally average it and compute AUC
        mean_tpr /= len(set(label_list))
        fpr_dict["macro"] = all_fpr
        tpr_dict["macro"] = mean_tpr
        roc_auc_dict["macro"] = auc(fpr_dict["macro"], tpr_dict["macro"])

        # 绘制所有类别平均的roc曲线
        plt.figure(figsize=(8, 8))
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
        for i, color in zip(range(len(set(label_list))), colors):
            plt.plot(fpr_dict[i], tpr_dict[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(class_indict[str(i)], roc_auc_dict[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw, label='Chance', color='red')
        plt.xlim([0.0, 1.01])
        plt.ylim([0.0, 1.01])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        # plt.savefig('multi_class_ROC.png')
        plt.show()


    def train_step(self, net, data_loader, optimizer, epoch, device):
        net.train()
        train_loss, train_acc, sampleNum = 0, 0, 0
        loss_function = nn.CrossEntropyLoss()
        optimizer.zero_grad()

        data_loader = tqdm(data_loader, file=sys.stdout, colour='red')
        for step, data in enumerate(data_loader):
            x, y = data
            sampleNum += x.shape[0]
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            predictions = net(x)
            loss = loss_function(predictions, y)
            train_loss += loss.item()
            train_acc += (torch.argmax(predictions, dim=1) == y).sum().item()
            loss.backward()
            optimizer.step()
            data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch, train_loss / (step + 1),
                                                                                   train_acc / sampleNum)
            return round(train_loss / (step + 1), 3), round(train_acc / sampleNum, 3)


    @torch.no_grad()
    def val_step(self, net, data_loader, epoch, device):

        loss_function = nn.CrossEntropyLoss()
        net.eval()
        val_acc, val_loss, sample_num = 0, 0, 0
        val_bar = tqdm(data_loader, file=sys.stdout, colour='red')
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
        return round(val_loss / (step + 1), 3), round(val_acc / sample_num, 3)


    def Predictor(self, net, valid_loader):
        try:
            json_file = open('./classes_indices.json', 'r')
            class_indict = json.load(json_file)
        except Exception as e:
            print(e)
            exit(-1)

        y_true, y_pred = [], []
        errors = 0
        net.eval()
        net.load_state_dict(torch.load(self.save_path))
        with torch.no_grad():
            for step, data in enumerate(valid_loader):
                x, y = data
                x, y = x.to(self.device), y.to(self.device)
                predictions = net(x)
                predictions = torch.argmax(
                    torch.softmax(predictions, dim=1), dim=1)
                for i in range(len(predictions)):
                    y_true.append(y[i].cpu())
                    y_pred.append(predictions[i].cpu())

        tests = len(y_true)
        for i in range(tests):
            pred_index = y_pred[i]
            true_index = y_true[i]
            if pred_index != true_index:
                errors += 1

        acc = (1 - errors / tests) * 100
        print(f'there were {errors} errors in {tests} tests for an accuracy of {acc:6.2f}%')

        ypred = np.array(y_pred)
        ytrue = np.array(y_true)

        f1score = f1_score(ytrue, ypred, average='weighted') * 100
        print(f'The F1-score was {f1score:.3f}')

        class_count = len(list(class_indict.values()))
        classes = list(class_indict.values())

        cm = confusion_matrix(ytrue, ypred)
        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, vmin=0, fmt='g', cmap='Blues', cbar=False)
        plt.xticks(np.arange(class_count) + .5, classes, rotation=0, fontsize=14)
        plt.yticks(np.arange(class_count) + .5, classes, rotation=0, fontsize=14)
        plt.xlabel("Predicted", fontsize=14)
        plt.ylabel("Actual", fontsize=14)
        plt.title("Confusion Matrix")

        plt.subplot(1, 2, 2)
        sns.heatmap(cm / np.sum(cm), annot=True, fmt='.1%')
        plt.xlabel('Predicted Label', fontsize=14)
        plt.xticks(np.arange(class_count) + .5, classes, rotation=0, fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        plt.yticks(np.arange(class_count) + .5, classes, rotation=0, fontsize=14)
        plt.show()

        clr = classification_report(y_true, y_pred, target_names=classes, digits=4)
        print("Classification Report:\n----------------------\n", clr)

    def plot_acc_loss(self, df):
        Epochs = list(range(self.epochs))
        tloss = df.loc[:, 'train_loss'].tolist()
        vloss = df.loc[:, 'val_loss'].tolist()
        tacc = df.loc[:, 'train_accuracy'].tolist()
        vacc = df.loc[:, 'val_accuracy'].tolist()

        plt.style.use('fivethirtyeight')
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
        axes[0].plot(Epochs, tloss, 'r', label='Training loss')
        axes[0].plot(Epochs, vloss, 'g', label='Validation loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epochs', fontsize=18)
        axes[0].set_ylabel('Loss', fontsize=18)
        axes[0].legend()

        axes[1].plot(Epochs, tacc, 'r', label='Training Accuracy')
        axes[1].plot(Epochs, vacc, 'g', label='Validation Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].set_xlabel('Epochs', fontsize=18)
        axes[1].set_ylabel('Accuracy', fontsize=18)
        axes[1].legend()
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    net = NeuralNetwork()
    net.fit()