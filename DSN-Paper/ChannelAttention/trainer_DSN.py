from __future__ import division
import torch
from torch.utils import data
from model.SCNN import SCNN
from utils.utils import generate_random_str
from utils.dataset_timeseries import load_UCR_data, get_timeseries_dataset, load_dataset_mul
from utils.TSC_data_loader import TSC_multivariate_data_loader
import argparse
import sparselearning
from sparselearning.core_kernel import Masking, CosineDecay, str2bool
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import LabelEncoder
from model.layers import *
import os
from model.SCNN import SCNN_with_CA

def train(args, model, train_loader, optimizer, epoch, mask=None, weights=None):
    train_loss = 0
    correct = 0
    n = 0
    model.train()
    for data_item in train_loader:
        im = data_item[0].type(torch.FloatTensor).cuda()
        # 如果输入为二维（batch, seq_len），增加通道维度
        if im.dim() == 2:
            im = im.unsqueeze(1)
        label = data_item[1].cuda()
        
        optimizer.zero_grad()
        outputs = model(im)
        loss = F.nll_loss(outputs, label.long())
        loss.backward()
        
        if mask is not None:
            mask.step()
        else:
            optimizer.step()
        
        train_loss += loss.item()
        pred = outputs.argmax(dim=1, keepdim=True)
        correct += pred.eq(label.view_as(pred)).sum().item()
        n += label.shape[0]
    
    cur_lr = optimizer.param_groups[0]['lr']
    # 每20个epoch打印一次日志
    if epoch % 20 == 0:
        print('Epoch: {}, Cur_lr: {:.5f}, Train Loss: {:.5f}, Train Acc: {}/{} {:.3f}'.format(
            epoch, cur_lr, train_loss / n, correct, n, 100. * correct / n))
    
    return train_loss / n

def evaluate(args, model, test_loader, weights=None):
    model.eval()
    test_loss = 0
    correct = 0
    n = 0
    
    with torch.no_grad():
        for data_item in test_loader:
            im = data_item[0].type(torch.FloatTensor).cuda()
            # 保证测试数据也有通道维度
            if im.dim() == 2:
                im = im.unsqueeze(1)
            label = data_item[1].cuda()
            output = model(im)
            test_loss += F.nll_loss(output, label.long(), reduction='sum').item()
            
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()
            n += label.shape[0]
    
    test_loss /= float(n)
    print('### Valid Loss: {:.5f}, Valid Acc: {}/{} {:.3f}'.format(
        test_loss, correct, n, 100. * correct / n))
    return 100. * correct / n

def main(args=None):
    print(args)
    torch.manual_seed(args.seed)
    
    dataset_name = os.path.split(args.root)[-1]
    if 'UCR_TS' in dataset_name:
        X_train, Y_train, X_test, Y_test, nb_classes = load_UCR_data(
            root=args.root, file_name=args.data, normalize_timeseries=None)
    elif 'UEA_TS_Archive' in dataset_name:
        X_train, Y_train, X_test, Y_test, nb_classes = TSC_multivariate_data_loader(args.root, args.data)
    else:
        X_train, Y_train, X_test, Y_test, nb_classes = load_dataset_mul(args.root, args.data)
    
    batch_size = max(int(min(X_train.shape[0] / 10, args.batch_size)), 2)
    trainloader, valloader = get_timeseries_dataset(
        batch_size=batch_size, x_train=X_train, y_train=Y_train,
        x_test=X_test, y_test=Y_test, n_worker=0)
    
    # 计算类别权重
    classes = np.unique(Y_train)
    le = LabelEncoder()
    y_ind = le.fit_transform(Y_train.ravel())
    recip_freq = len(Y_train) / (len(le.classes_) * np.bincount(y_ind).astype(np.float64))
    class_weight = recip_freq[le.transform(classes)]
    print('Class weights: ', class_weight)
    
    # 构造基础 SCNN 模型，然后用 SCNN_with_CA 封装
    base_model = SCNN(
        c_in=X_train.shape[1],
        c_out=nb_classes,
        nf=args.ch_size,
        depth=args.depth,
        kernel=args.k_size,
        pad_zero=args.pad_zero
    )
    model = SCNN_with_CA(base_model)
    model.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, 1e-4, last_epoch=-1)
    
    mask = None
    if args.sparse:
        decay = CosineDecay(args.death_rate, len(trainloader) * args.epochs * 0.8)
        mask = Masking(optimizer, death_rate=args.death_rate, death_mode=args.death,
                       death_rate_decay=decay, growth_mode=args.growth,
                       redistribution_mode=args.redistribution,
                       train_loader=trainloader, args=args)
        mask.add_module(model, sparse_init=args.sparse_init, density=args.density)
    
    train_loss = 10.0
    random_str = generate_random_str(10)
    save_path = None
    for epoch in range(1, args.epochs + 1):
        output = train(args, model, trainloader, optimizer, epoch, mask, weights=class_weight)
        lr_scheduler.step()
        
        if epoch >= args.epochs * 0.8 and args.sparse:
            mask.death_decay_update(decay_flag=False)
        
        # 保存模型但不打印保存提示
        if train_loss >= output:
            save_path = 'models_save_app/DSN_sort_%s_%s_%s_%s.pth' % (
                args.data, args.density, args.c_size, random_str)
            train_loss = output
            torch.save(model.state_dict(), save_path)
    
    print('Testing model')
    model.load_state_dict(torch.load(save_path))
    val_acc = evaluate(args, model, valloader, weights=class_weight)
    print('### data name: {}, best_val_acc {:.3f}'.format(args.data, val_acc))
    
    return args.data, val_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N')
    parser.add_argument('--test-batch-size', type=int, default=16, metavar='N')
    parser.add_argument('--multiplier', type=int, default=1, metavar='N')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M')

    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=17, metavar='S')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N')
    parser.add_argument('--optimizer', type=str, default='admW')
    parser.add_argument('--root', type=str, default='data/MultiVariate')

    # 默认数据集设为 egg2
    parser.add_argument('--data', type=str, default='daily_sport')


    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--decay_frequency', type=int, default=25000)
    parser.add_argument('--l1', type=float, default=0.0)
    parser.add_argument('--fp16', action='store_true', help='Run in fp16 mode.')
    parser.add_argument('--valid_split', type=float, default=0.1)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=1.0e-3)
    parser.add_argument('--save-features', action='store_true')
    parser.add_argument('--max-threads', type=int, default=0)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--ch_size', type=int, default=47)
    parser.add_argument('--k_size', type=int, default=39)
    parser.add_argument('--pad_zero', type=str2bool, default=False)
    # sparselearning 会自动添加 --c_size 参数
    sparselearning.core_kernel.add_sparse_args(parser)
    
    args = parser.parse_args()
    args.density = 0.9
    args.death_rate = 0.2
    
    # 只运行一个数据集，由命令行参数 --data 指定（默认 egg2）
    main(args)
