import sys
import torch, argparse, os, time
from split_data import read_split_data
from utils import MyDataset, train, val_step, Plot_ROC, Predictor
from torch.utils.data import DataLoader
from config import device, weight_decay, data_transform
from ResNet_Swin import NeuralNetwork as create_model
from torch.optim.lr_scheduler import StepLR


def main(args):
    if os.path.exists("./ResNet_Swin") is False:
        os.makedirs("./ResNet_Swin")

    if os.path.exists("./save_images") is False:
        os.makedirs("./save_images")

    train_image_path, train_image_label, val_image_path, val_image_label, class_indices = read_split_data(args.data_path)

    # 实例化训练数据集
    train_dataset = MyDataset(train_image_path, train_image_label, data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataset(val_image_path, val_image_label, data_transform["valid"])

    batch_size = args.batch_size

    sys_name = sys.platform
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0,
              8]) if 'linux' in sys_name.lower() else 0  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=0,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=0,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.33)

    scalar = torch.cuda.amp.GradScaler()
    # print('Start Training!!!')
    # start_time = time.time()

    if args.resume:
        path_checkpoint = f"./ResNet_Swin/model_{args.epochs - 1}.pth"  # 断点路径
        checkpoint = torch.load(path_checkpoint)  # 加载断点

        model.load_state_dict(checkpoint['model'])  # 加载模型可学习参数
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']

        if scalar:
            scalar.load_state_dict(checkpoint["scalar"])

        print(f'from {start_epoch} epoch starts training!!!')

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train(model, train_loader, optimizer, device, epoch, scalar)

        # validate
        val_loss, val_acc = val_step(model, val_loader, device, epoch)

        lr_scheduler.step()

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if scalar:
            save_file["scalar"] = scalar.state_dict()
        torch.save(save_file, "./ResNet_Swin/model_{}.pth".format(epoch))

        # torch.save(model.state_dict(), "./ResNet_Swin/model_{}.pth".format(epoch))

    # end_time = time.time()
    # print(f'Use {round(end_time - start_time, 3)}seconds')
    # print('Finished Training!!!')

    Predictor(model, val_loader, f"./ResNet_Swin/model_{args.epochs - 1}.pth", device)
    Plot_ROC(model, val_loader, f"./ResNet_Swin/model_{args.epochs - 1}.pth", device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)  # 类别个数
    parser.add_argument('--epochs', type=int, default=2)  # 训练次数
    parser.add_argument('--batch-size', type=int, default=32)  # batch_size为32 内存占用18G 显存6G
    parser.add_argument('--lr', type=float, default=0.0003)  # learning_rate

    # 数据集所在根目录
    parser.add_argument('--data-path', type=str, default=r"D:/flower_data")
    # 断点续训
    parser.add_argument('--resume', type=bool, default=True)
    # 预训练权重路径，如果不想载入就设置为空字符
    # parser.add_argument('--weights', type=str, default='', help='initial weights path')

    # 是否冻结权重 如果只想训练最后一层head 需要改成True
    # parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()

    main(opt)
