from torch.optim.lr_scheduler import StepLR
from ResMLP_models import resmlp_12 as net
import sys
import torch, argparse, os, time
from split_data import read_split_data
from utils import train_step, val_step, Plot_ROC
from my_dataset import MyDataset
from torch.utils.data import DataLoader
from config import device, weight_decay, data_transform
import warnings

warnings.filterwarnings('ignore')

def main(args):

    if os.path.exists("./ResMLP_Model") is False:
        os.makedirs("./ResMLP_Model")

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


    model = net(num_classes=args.num_classes).to(device)
    
    """
    如果要使用迁移学习，调用预训练权重，则需要使用model.reset_classifier(num_classes=args.num_classes)
    """

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.33)

    scalar = torch.cuda.amp.GradScaler() if args.scalar else None

    resume = os.path.exists(f'./ResMLP_Model/model.pth')

    if resume:
        path_checkpoint = f"./ResMLP_Model/model.pth"  # 断点路径
        checkpoint = torch.load(path_checkpoint)  # 加载断点

        model.load_state_dict(checkpoint['model'])  # 加载模型可学习参数
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数

        if scalar:
            scalar.load_state_dict(checkpoint["scalar"])


    print('Start Training!!!')

    start_time = time.time()
    initial_accuracy = 0.0

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

        if val_acc > initial_accuracy:

            save_file = {"model": model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "lr_scheduler": lr_scheduler.state_dict(),
                         "args": args}
            if scalar:
                save_file["scalar"] = scalar.state_dict()
            torch.save(save_file, "./ResMLP_Model/model.pth")

            initial_accuracy = val_acc


    end_time = time.time()
    print(f'Use {round(end_time - start_time, 3)}seconds')
    print('Finished Training!!!')

    Plot_ROC(model, val_loader, "./ResMLP_Model/model.pth", device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default=r"D:/flower_data") # data directory
    parser.add_argument('--num_classes', type=int, default=5) # number classes
    parser.add_argument('--epochs', type=int, default=5) # train epochs
    parser.add_argument('--batch-size', type=int, default=32) # batch_size为32  内存占用19G 显存3G
    parser.add_argument('--lr', type=float, default=0.0003) # learning_rate
    parser.add_argument('--weight-decay', type=float, default=1e-5)  # weight_decay

    # checkpoint
    # parser.add_argument('--resume', type=bool, default=True)

    # amp
    parser.add_argument('--scalar', type=bool, default=True)

    opt = parser.parse_args()

    main(opt)
