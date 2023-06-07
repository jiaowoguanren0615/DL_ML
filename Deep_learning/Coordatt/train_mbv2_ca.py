from torch.optim.lr_scheduler import StepLR
from mbv2_ca import MBV2_CA
from resnet import net
import sys
import torch, argparse, os, time
from split_data import read_split_data
from utils import train_step, val_step
from my_dataset import MyDataset
from torch.utils.data import DataLoader
from config import device, weight_decay, data_transform
import warnings



warnings.filterwarnings('ignore')

def main(args):

    if os.path.exists("./CoordAttention") is False:
        os.makedirs("./CoordAttention")

    train_image_path, train_image_label, val_image_path, val_image_label, class_indices = read_split_data(args.data_path)

    # train dataset
    train_dataset = MyDataset(train_image_path, train_image_label, data_transform["train"])

    # valid dataset
    val_dataset = MyDataset(val_image_path, val_image_label, data_transform["valid"])

    batch_size = args.batch_size

    sys_name = sys.platform
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]) if 'linux' in sys_name.lower() else 0  # number of workers
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

    # model = MBV2_CA(num_classes=args.num_classes).to(device)

    model = net.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.33)

    scalar = torch.cuda.amp.GradScaler() if args.scalar else None

    resume = os.path.exists(f'./CoordAttention/model_{args.epochs - 1}.pth')
    if resume:
        path_checkpoint = f"./CoordAttention/model_{args.epochs - 1}.pth"  # 断点路径
        checkpoint = torch.load(path_checkpoint)  # 加载断点

        model.load_state_dict(checkpoint['model'])  # 加载模型可学习参数
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']

        if scalar:
            scalar.load_state_dict(checkpoint["scalar"])

        print(f'from {start_epoch} epoch starts training!!!')

    print('Start Training!!!')

    start_time = time.time()

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_step(net=model,
                                           optimizer=optimizer,
                                           data_loader=train_loader,
                                           device=device,
                                           epoch=epoch,
                                           scalar=scalar)

        # validate
        val_loss, val_acc = val_step(net=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        lr_scheduler.step()

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if scalar:
            save_file["scalar"] = scalar.state_dict()
        torch.save(save_file, "./CoordAttention/model_{}.pth".format(epoch))

        # torch.save(model.state_dict(), "./CoordAttention/model_{}.pth".format(epoch))

    end_time = time.time()
    print(f'Use {round(end_time - start_time, 3)}seconds')
    print('Finished Training!!!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5) # number classes
    parser.add_argument('--epochs', type=int, default=5) # train epochs
    parser.add_argument('--batch-size', type=int, default=32) # batch_size为32 内存占用14G 显存4G
    parser.add_argument('--lr', type=float, default=0.0003) # learning_rate
    parser.add_argument('--weight-decay', type=float, default=1e-5)  # weight_decay

    # data directory
    parser.add_argument('--data-path', type=str, default=r"D:/flower_data")

    # pretrain weights directory
    # parser.add_argument('--weights', type=str, default='', help='initial weights path')

    # chill weights
    # parser.add_argument('--freeze-layers', type=bool, default=False)

    # checkpoint
    # parser.add_argument('--resume', type=bool, default=True)

    # amp
    parser.add_argument('--scalar', type=bool, default=True)
    # device
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()

    main(opt)