"""
该脚本主要解决多分类机器学习场景下ROC绘制
"""

import re
import os, json
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import warnings
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import auc, f1_score, classification_report, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from numpy import interp
from itertools import cycle


warnings.filterwarnings('ignore')


class Model():
    def __init__(self,
                 data_root,
                 target_name,
                 drop_columns=None,
                 model_name=None,
                 norm=None,
                 data_reduction=False,
                 plot_every_class_distribution=True,
                 plot_feature_importance=True,
                 predict=True,
                 plot_roc=True):

        """
        :param data_root: Your data root path
        :param target_name: The target column's name in your data file
        :param drop_columns: Choose the column what you want to drop in raw data (type: list) eg:['A', 'B'] or ['A']
        :param model_name: Choose your model
        :param norm: Choose your data Pre-process idea
        :param data_reduction: Use data_reduction (eg: PCA)
        :param plot_every_class_distribution: Do you want to plot every classes distribution in a bar_plot? (default False)
        :param plot_feature_importance: Do you need to plot the feature_importance? (default False) 如果需要画特征重要性 则必须使用LGBM算法
        :param predict: Do you need predict and plot confusion_matrix? (default True)
        :param plot_roc: Do you want to plot ROC curve for every classes (default True)
        """

        self.root = data_root

        assert os.path.exists(self.root), 'Your data_path is wrong!'

        self.target_name = target_name
        self.model_name = model_name
        self.drop_columns = drop_columns
        self.norm = norm
        self.data_reduction = data_reduction
        self.plot_every_class_distribution = plot_every_class_distribution
        self.plot_feature_importance = plot_feature_importance
        self.predict = predict
        self.plot_roc = plot_roc

    # 读取数据
    def read_data(self):
        # Read your data file 通过判断文件后缀名来具体使用哪种api来读取数据
        if re.findall('.csv', self.root):
            dfMerge = pd.read_csv(self.root)
        elif re.findall('.txt', self.root):
            dfMerge = pd.read_table(self.root)
        elif re.findall('.xlsx|.xls', self.root):
            dfMerge = pd.read_excel(self.root)
        else:
            raise ValueError('Your file does not belongs to one of csv txt xlsx, please modify the process reading data by yourself')

        classes = dfMerge[self.target_name].unique()
        self.target_dict = {k: v for k, v in enumerate(classes)}

        class_indices = {k: v for v, k in enumerate(classes)}
        json_str = json.dumps({v: k for k, v in class_indices.items()}, indent=4)
        with open('./classes_indices.json', 'w') as json_file:
            json_file.write(json_str)

        dfMerge[self.target_name] = dfMerge[self.target_name].map(class_indices)

        if self.drop_columns is not None:
            dfMerge = dfMerge.drop(self.drop_columns, axis=1)

        every_class_num = []
        for colName in dfMerge[self.target_name].unique():
            every_class_num.append(dfMerge[dfMerge[self.target_name] == colName].shape[0])

        if self.plot_every_class_distribution:
            plt.bar(range(len(classes)), every_class_num, align='center')
            plt.xticks(range(len(classes)), classes)

            for i, v in enumerate(every_class_num):
                plt.text(x=i, y=v + 5, s=str(v), ha='center')

            plt.xlabel('data class')
            plt.ylabel('count')
            plt.title('class distribution')
            plt.show()
        return dfMerge

    def run(self):
        dfMerge, X_train, X_test, y_train, y_test = self.data_process()
        if 'LGR' in self.model_name.upper():
            clf = LogisticRegression()
        elif 'SVM' in self.model_name.upper():
            clf = SVC()
        else:
            clf = LGBMClassifier()

        model = clf.fit(X_train, y_train)

        print('The score is : ', (model.predict(X_test) == y_test).sum() / y_test.shape[0] * 100)

        if self.plot_feature_importance and self.model_name == 'LGBM':
            self.plotImp(model, dfMerge)

        if self.predict:
            self.Predictor(model, X_test, y_test)

        if self.plot_roc:
            self.Plot_ROC(dfMerge)

    # 画出特征重要性图 由大到小
    def plotImp(self, model, dfMerge, num=3, fig_size=(20, 10)):
        dfMerge = dfMerge.drop([self.target_name], axis=1)
        feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': dfMerge.columns})
        plt.figure(figsize=fig_size)
        # sns.set(font_scale=3)
        sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
        plt.title('LightGBM Features (avg over folds)')
        plt.tight_layout()
        plt.savefig('lgbm_importances-01.png')
        plt.show()

    # 数据预处理过程 包括特征降维 标准化(归一化处理) 如果需要使用PCA，则不要画特征重要性 否则会报错 因为特征数量不同了 无法构造dataframe
    def data_process(self):
        dfMerge = self.read_data()
        X = np.array(dfMerge.drop([self.target_name], axis=1))
        y = np.array(dfMerge[self.target_name])
        if self.data_reduction:
            X = PCA(n_components=int(X.shape[1] // 4)).fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

        if self.norm:
            sc = self.norm
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
        return dfMerge, X_train, X_test, y_train, y_test

    # 预测并画出混淆矩阵
    def Predictor(self, model, X_test, y_test):
        try:
            json_file = open('./classes_indices.json', 'r')
            class_indict = json.load(json_file)
        except Exception as e:
            print(e)
            exit(-1)

        y_true = y_test
        y_pred = model.predict(X_test)
        tests = len(y_pred)
        errors = 0

        for i in range(tests):
            if y_true[i] != y_pred[i]:
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

    # 画出ROC曲线
    def Plot_ROC(self, dfMerge):

        try:
            json_file = open('./classes_indices.json', 'r')
            class_indict = json.load(json_file)
        except Exception as e:
            print(e)
            exit(-1)


        X = dfMerge.drop([self.target_name], axis=1)
        y = np.array(dfMerge[self.target_name])

        n_classes = len(dfMerge[self.target_name].unique())
        y = label_binarize(y, classes=list(range(n_classes)))

        lr = LGBMClassifier()
        classifier = OneVsRestClassifier(lr)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)

        y_score = classifier.fit(X_train, y_train).predict(X_test)

        # 计算每一类的ROC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area（方法二）
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Compute macro-average ROC curve and ROC area（方法一）
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        lw = 2
        plt.figure(figsize=(12, 12))
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(class_indict[str(i)], roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw, label='Chance', color='red')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.savefig('Roc.png')
        plt.show()



if __name__ == '__main__':
    model = Model(data_root='./milknew.csv', target_name='Grade', drop_columns=None, model_name='LGBM', norm=StandardScaler())
    model.run()
