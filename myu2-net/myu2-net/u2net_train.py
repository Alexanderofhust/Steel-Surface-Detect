import os
import datetime
import torch
import numpy as np
from tqdm import tqdm
from src.u2net import U2NET, U2NETP
from src.seg_dataset import U2netSegDataset
from torch.utils.data import DataLoader

# 参考u2net源码loss的设定
bce_loss = torch.nn.BCELoss(reduction='mean')


def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    # print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (
    # loss0.data.item(), loss1.data.item(), loss2.data.item(), loss3.data.item(), loss4.data.item(), loss5.data.item(),
    # loss6.data.item()))

    return loss0, loss


def load_data(img_folder, mask_folder, batch_size, num_workers, input_size):
    """
    :param img_folder: 图片保存的fodler
    :param mask_folder: mask保存的fodler
    :param batch_size: batch_size的设定
    :param num_workers: 数据加载cpu核心数
    :param input_size: 模型输入尺寸
    :return:
    """
    train_dataset = U2netSegDataset(img_dir=os.path.join(os.getcwd(),img_folder,'images', 'train'),
                                    mask_dir=os.path.join(os.getcwd(),mask_folder,'masks', 'train'),
                                    input_size=input_size)

    val_dataset = U2netSegDataset(img_dir=os.path.join(os.getcwd(),img_folder, 'images', 'test'),
                                  mask_dir=os.path.join(os.getcwd(),mask_folder,'masks', 'test'),
                                  input_size=input_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


def train_model(epoch_nums, cuda_device, model_save_dir):
    """
    :param epoch_nums: 训练总的epoch
    :param cuda_device: 指定gpu训练
    :param model_save_dir: 模型保存folder
    :return:
    """
    current_time = datetime.datetime.now()
    current_time = datetime.datetime.strftime(current_time, '%Y-%m-%d-%H:%M')
    model_save_dir = os.path.join(os.getcwd(), model_save_dir)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    else:
        pass

    device = torch.device('cuda:0')  ##这里gpu
    train_loader, val_loader = load_data(img_folder='dataset',
                                         mask_folder='dataset',
                                         batch_size=32,
                                         num_workers=10,
                                         input_size=(160, 160))

    # input 3-channels, output 1-channels
    net = U2NET(3, 1)
    # net = U2NETP(3, 1)

    # if torch.cuda.device_count() > 1:
    #     net = torch.nn.DataParallel(net, device_ids=[6, 7])
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    for epoch in range(0, epoch_nums):
        run_loss = list()
        run_tar_loss = list()
        run_iou = list()

        net.train()
        for i, (inputs, gt_masks) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            inputs = inputs.type(torch.FloatTensor)
            gt_masks = gt_masks.type(torch.FloatTensor)
            inputs, gt_masks = inputs.to(device), gt_masks.to(device)

            d0, d1, d2, d3, d4, d5, d6 = net(inputs)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, gt_masks)

            loss.backward()
            optimizer.step()

            run_loss.append(loss.item())
            run_tar_loss.append(loss2.item())

            # 计算IoU
            iou = calculate_iou(d0, gt_masks)
            run_iou.append(iou)

            del d0, d1, d2, d3, d4, d5, d6, loss2, loss

        print("--Train Epoch:{}--".format(epoch))
        print("--Train run_loss:{:.4f}--".format(np.mean(run_loss)))
        print("--Train run_tar_loss:{:.4f}--".format(np.mean(run_tar_loss)))
        print("--Train run_iou:{:.4f}--\n".format(np.mean(run_iou)))

        if epoch % 5 == 0:
            checkpoint_name = 'checkpoint_' + str(epoch) + '_' + str(np.mean(run_loss)) + '.pth'
            torch.save(net.state_dict(), os.path.join(model_save_dir, '1.pth'))
            print("--model saved:{}--".format(checkpoint_name))

def calculate_iou(pred, target):
    """
    计算两个二值张量的IoU值。

    参数:
    pred -- 预测的二值张量
    target -- 真实的二值张量

    返回:
    iou -- IoU值
    """
    # 将预测结果转换为二值
    pred = (pred > 0.5).float()

    # 计算交集
    intersection = (pred * target).sum()

    # 计算并集
    union = pred.sum() + target.sum() - intersection

    # 防止除以零
    if union == 0:
        return 0.0

    # 计算IoU
    iou = intersection / union

    return iou.item()

if __name__ == '__main__':
    train_model(epoch_nums=10    , cuda_device='cuda:1',
                model_save_dir='backup')

