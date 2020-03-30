import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from utils.MyDataset import TrainDataSet, TestDataset, get_train_trans, get_test_trans
from utils.logger import CsvLogger
from models.MyModel import PlantModel, save_checkpoints
from models.MyLoss import DenseCrossEntropy
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import pandas as pd
import argparse
import tqdm
import numpy as np


def parser():
    parser = argparse.ArgumentParser(description='plant pathology')
    # data
    parser.add_argument('--img_path', type=str, default='data/images/')
    parser.add_argument('--train_df', type=str, default='data/train.csv')
    parser.add_argument('--submit_df', type=str, default='data/sample_submission.csv')
    parser.add_argument('--n_flod', type=int, default=5)
    # model
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_class', type=int, default=4)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--num_epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_save_path', type=str, default='checkpoints/')
    # log
    parser.add_argument('--log_path', type=str, default='logs/')
    parser.add_argument('--log_file', type=str, default='results.csv')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parser()
    print(args)
    all_data = pd.read_csv(args.train_df)
    data = all_data.iloc[:, 0].values
    targets = all_data.iloc[:, [1, 2, 3, 4]].values
    train_y = targets[:, 0] + targets[:, 1] * 2 + targets[:, 2] * 3
    skf = StratifiedKFold(args.n_flod, shuffle=True, random_state=42)
    # ==========logger==========
    logger = CsvLogger(args.log_path, args.log_file)
    # ==========model==========
    device = torch.device('cuda:0')
    model: nn.Module = PlantModel(pretrained=True, num_class=args.num_class).to(device)
    criterion = DenseCrossEntropy()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.98)
    BEST_LOSS = 99
    # ==========data==========
    off_pred = np.zeros((data.shape[0], 4), dtype=np.float)
    for i_fold, (train_index, val_index) in enumerate(skf.split(data, train_y)):
        X_train, X_val = data[train_index], data[val_index]
        y_train, y_val = targets[train_index], targets[val_index]
        train_dataset = TrainDataSet(args.img_path, X_train, y_train, transformation=get_train_trans(args.size))
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=train_dataset.collate_fn)
        val_dataset = TrainDataSet(args.img_path, X_val, y_val, transformation=get_test_trans(args.size))
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=train_dataset.collate_fn)
        for epoch in range(args.num_epoch):
            train_loss = 0
            val_loss = 0
            print('==========training===========')
            model.train()
            for index, (train_imgs, train_targets) in enumerate(tqdm.tqdm(train_dataloader, desc='training')):
                train_imgs = train_imgs.to(device)
                train_targets = train_targets.to(device)
                train_output = model(train_imgs)
                loss = criterion(train_output, train_targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            print('==========valing===========')
            model.eval()
            val_preds = None  # 记录每个i_fold的预测类别
            for index, (val_imgs, val_targets) in enumerate(tqdm.tqdm(val_dataloader, desc='valing')):
                val_imgs = val_imgs.to(device)
                val_targets = val_targets.to(device)
                with torch.no_grad():
                    val_output = model(val_imgs)
                    loss = criterion(val_output, val_targets)
                val_loss += loss.item()
                preds = torch.softmax(val_output, dim=1).data.cpu()
                if val_preds == None:
                    val_preds = preds
                else:
                    val_preds = torch.cat([val_preds, preds], dim=0)

            if val_loss / len(val_dataloader) < BEST_LOSS:
                is_best = True
                BEST_LOSS = val_loss / len(val_dataloader)
            else:
                is_best = False

            print('i_fold:{},epoch:{},train_loss:{},val_loss{},lr:{}'.format(
                i_fold, epoch, train_loss / len(train_dataloader), val_loss / len(val_dataloader), scheduler.get_lr()[0]
            ))
            logger.write({'epoch': epoch + i_fold * args.num_epoch, 'train_loss': train_loss / len(train_dataloader),
                          'val_loss': val_loss / len(val_dataloader)})
            logger.plot_progress()
            save_checkpoints({'epoch': epoch + i_fold * args.num_epoch,
                              'state_dict': model.state_dict(),
                              'optimizer': optimizer.state_dict()}, is_best=is_best,
                             fpath=args.weight_save_path, fname='model_{}.pth'.format(epoch + i_fold * args.num_epoch))
            scheduler.step()
        off_pred[val_index, :] = val_preds
    score = roc_auc_score(y_true=targets, y_score=off_pred, average='macro')
    print('5-Folds CV score: {:.4f}'.format(score))
    print('==========testing===========')
    test_data = pd.read_csv(args.submit_df)
    test_dataset = TestDataset(args.img_path, test_data.iloc[:, 0].values, transformation=get_test_trans(args.size))
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn)
    test_pred = None
    for index, (test_imgs) in enumerate(tqdm.tqdm(test_dataloader, desc='valing')):
        test_imgs = test_imgs.to(device)
        model.eval()
        with torch.no_grad():
            test_output = model(test_imgs)
        if test_pred is None:
            test_pred = test_output.data.cpu()
        else:
            test_pred = torch.cat((test_pred, test_output.data.cpu()), dim=0)
    test_data[['healthy', 'multiple_diseases', 'rust', 'scab']] = torch.softmax(test_pred, dim=1)
    test_data.to_csv('results/result.csv', index=False)
    print('finish')
