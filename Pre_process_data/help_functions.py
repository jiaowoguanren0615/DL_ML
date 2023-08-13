"""
该脚本主要包括数据科学 机器学习以下几个步骤：
1 数据读取部分 数据集划分
2 数据预处理 标准化(归一化) 特征降维
3 算法选择 模型搭建
4 特征重要性筛选 (也可以用筛选之后的特征重新搭建模型 并且筛选之后的特征数量一定少于原特征数量 也能起到降维做用)
5 预测 评测指标 混淆矩阵 ROC曲线绘制

注：该脚本未实现数据缺失值填充，如果自己的数据有缺失值，则需要在read_data函数中增加dfMerge.fillna相关代码
"""

import re
import os, json
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import warnings
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import auc, f1_score, classification_report, confusion_matrix, RocCurveDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

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

        if self.plot_feature_importance:
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
        X = dfMerge.drop([self.target_name], axis=1)
        y = np.array(dfMerge[self.target_name])

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        lr = LGBMClassifier()

        pipe_lr = Pipeline([('scaler', StandardScaler()), ('clf', lr)])

        tprs = []
        aucs = []

        mean_fpr = np.linspace(0, 1, 100)

        fig, ax = plt.subplots(figsize=(8, 8))
        models = []
        y_Actual, y_Predicted = [], []
        for i, (train, test) in enumerate(kf.split(X, y)):
            # print(X[train], y[train])
            X_train, X_test = X.iloc[train], X.iloc[test]
            y_train, y_test = y[train], y[test]
            pipe_lr.fit(X_train, y_train)
            viz = RocCurveDisplay.from_estimator(pipe_lr, X_test, y_test,
                                 name='ROC fold {}'.format(i),
                                 alpha=0.3, lw=1, ax=ax)
            models.append(pipe_lr)
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)
            y_Actual.extend(y_test)
            y_Predicted.extend(pipe_lr.predict(X_test))

        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                label='Chance', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color='b',
                label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')

        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
               title="Receiver operating characteristic example")
        ax.legend(loc="lower right")
        plt.savefig('Lgbm.Roc.png')
        plt.show()


if __name__ == '__main__':
    model = Model(data_root='./breast-cancer.csv', target_name='diagnosis', drop_columns=['id'], model_name='LGBM', norm=StandardScaler())
    model.run()
