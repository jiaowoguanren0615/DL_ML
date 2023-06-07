import sys
import torch, argparse, os, time
from split_data import read_split_data, train_one_epoch, val_step, Plot_ROC, Predictor
from my_dataset import MyDataset
from torch.utils.data import DataLoader
from config import device, weight_decay, data_transform
from model import convnext_tiny as create_model
from torch.optim.lr_scheduler import StepLR


def main(args):
    if os.path.exists("./ConvNext_weights") is False:
        os.makedirs("./ConvNext_weights")

    if os.path.exists("./save_images") is False:
        os.makedirs("./save_images")
    train_image_path, train_image_label, val_image_path, val_image_label, class_indices = read_split_data(args.data_path)

    # 实例化训练数据集
    train_dataset = MyDataset(train_image_path, train_image_label, data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataset(val_image_path, val_image_label, data_transform["valid"])

    batch_size = args.batch_size

    sys_name = sys.platform
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]) if 'linux' in sys_name.lower() else 0  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.33)

    scalar = torch.cuda.amp.GradScaler()

    print('Start Training!!!')
    start_time = time.time()
    for epoch in range(args.epochs):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, device, epoch, scalar)

        valid_loss, valid_accuracy = val_step(model, val_loader, device, epoch)

        lr_scheduler.step()

        torch.save(model.state_dict(), "./ConvNext_weights/model_{}.pth".format(epoch))

    end_time = time.time()
    print(f'Use {round(end_time - start_time, 3)}seconds')
    print('Finished Training!!!')
    Predictor(model, val_loader, f"./ConvNext_weights/model_{args.epochs - 1}.pth", device)
    Plot_ROC(model, val_loader, f"./ConvNext_weights/model_{args.epochs - 1}.pth", device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)  # 类别个数
    parser.add_argument('--epochs', type=int, default=2)  # 训练次数
    parser.add_argument('--batch-size', type=int, default=32)  # batch_size为32 内存占用18G 显存6G
    parser.add_argument('--lr', type=float, default=0.0003)  # learning_rate

    # 数据集所在根目录
    parser.add_argument('--data-path', type=str, default=r"D:/flower_data")

    # 预训练权重路径，如果不想载入就设置为空字符
    # parser.add_argument('--weights', type=str, default='', help='initial weights path')

    # 是否冻结权重 如果只想训练最后一层head 需要改成True
    # parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()

    main(opt)