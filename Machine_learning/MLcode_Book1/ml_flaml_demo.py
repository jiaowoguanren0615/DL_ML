from sklearn.model_selection import train_test_split
from flaml import AutoML
from sklearn.datasets import load_breast_cancer


dfData = load_breast_cancer()
X = dfData.data
y = dfData.target

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = AutoML()

model.fit(X_train, y_train, task='classification', metric='accuracy', time_budget=10)


print('Best ML Model:', model.best_estimator)
print('Best hyperparmeter config:', model.best_config)
print('Best Accuracy on validation data: %f'%(1 - model.best_loss))
print('Training duration of best run: %f s'%(model.best_config_train_time))