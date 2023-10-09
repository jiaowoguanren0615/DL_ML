import os
import torch, json
import matplotlib.pyplot as plt
from model import make_model
import numpy as np
from PIL import Image
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import seaborn as sns
from config import data_transform, save_path, device


@torch.no_grad()
def predictor(testloader):

    try:
        json_file = open('./classes_indices.json', 'r')
        class_indict = json.load(json_file)
    except Exception as e:
        print(e)
        exit(-1)

    errors = 0
    y_true, y_pred = [], []
    net = make_model(num_classes=5).to(device)

    assert os.path.exists(save_path), 'Can not find the file of your model_weights'
    net.load_state_dict(torch.load(save_path)['model'])
    
    net.eval()
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        preds = torch.argmax(torch.softmax(net(images), dim=1), dim=1)
        for i in range(len(preds)):
            y_true.append(labels[i].cpu())
            y_pred.append(preds[i].cpu())

    tests = len(y_pred)
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
    print(f'F1-score was {f1score:.3f}')

    class_count = len(list(class_indict.values()))
    classes = list(class_indict.values())
    
    cm = confusion_matrix(ytrue, y_pred)

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, vmin=0, fmt='g', cmap='Blues', cbar=False)
    plt.xticks(np.arange(class_count) + .5, classes, rotation=90, fontsize=14)
    plt.yticks(np.arange(class_count) + .5, classes, rotation=0, fontsize=14)
    plt.xlabel("Predicted", fontsize=14)
    plt.ylabel("Actual", fontsize=14)
    plt.title("Confusion Matrix")

    plt.subplot(1, 2, 2)
    sns.heatmap(cm / np.sum(cm), annot=True, fmt='.1%')
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()

    clr = classification_report(y_true, y_pred, target_names=classes, digits=4)
    print("Classification Report:\n----------------------\n", clr)
    
    return f1score


@torch.no_grad()
def predict_single_image():
        
    img_transform = data_transform['valid']

    # load image
    img_path = "./rose.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = img_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './classes_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = make_model(num_classes=5).to(device)
    # load model weights

    assert os.path.exists(save_path), "file: '{}' dose not exist.".format(save_path)
    model.load_state_dict(torch.load(save_path, map_location=device)['model'])

    model.eval()
    # predict class
    output = torch.squeeze(model(img.to(device))).cpu()
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    # plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    # plt.show()
