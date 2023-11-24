import argparse
import random
from model.main_model import main_model_ood
from utils.early_stopping import EarlyStopping
from data_load import datautils
import gc
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
import torch.nn.functional as F

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time Series to Graph for Classification')

    parser.add_argument('--data', type=str, default='ArticularyWordRecognition', help='name of data')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--max_epoch', type=int, default=200, help='max epoch')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--r1', type=float, default=0.5,
                        help='the ratio of nodes to choose as neighbors in intra-correlation extraction')
    parser.add_argument('--r2', type=float, default=0.7,
                        help='the ratio of nodes to choose as neighbors in inter-correlation extraction')
    parser.add_argument('--seg_len', type=int, help='the length of each segment')
    parser.add_argument('--gpu', type=int, default=0,
                        help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--d_model1', type=int, default=96,
                        help='the hidden dim of feature after building graph')
    parser.add_argument('--d_model2', type=int, default=32,
                        help='the dimension of feature after dimensional graph')
    parser.add_argument('--seed', type=int, default=20, help='random seed')
    parser.add_argument('--k_hops', type=int, default=3, help='the number of neighbors')
    parser.add_argument('--pe_ratio', type=float, default=0.8, help='the ratio of positional encoding dimension')
    parser.add_argument('--patience', type=int, default=15, help='patience to end training, default is 15')
    parser.add_argument('--max_dim', type=int, default=50, help='max dimension to avoid out of memory')
    parser.add_argument('--alpha', type=float ,default=1, help='the weight of domain cross entropy loss')
    parser.add_argument('--save_path', type=str, default='model_save/ood', help='the path to save model')
    parser.add_argument('--result_path', type=str, default='result.txt', help='the path to save results')

    args = parser.parse_args()
    for k, v in sorted(vars(args).items()):
        print(k, '=', v, end='. ')

    setup_seed(args.seed)
    if torch.cuda.is_available():
        print('GPU available.')
        device = torch.device('cuda:{}'.format(args.gpu))
    else:
        device = 'cpu'

    if args.data in ['DuckDuckGeese', 'FaceDetection', 'InsectWingbeat', 'JapaneseVowels']:
        train_dataset, val_dataset, test_dataset = datautils.load_UEA(args.data, args.max_dim)
    else:
        train_dataset, val_dataset, test_dataset = datautils.load_UEA_csv(args.data, args.max_dim)
    train_data_num = train_dataset.data_x.shape[0]
    val_data_num = val_dataset.data_x.shape[0]
    test_data_num = test_dataset.data_x.shape[0]

    train_ts = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_ts = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_ts = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    if train_dataset.merge_dim != 0:
        args.seg_len = args.seg_len * train_dataset.merge_dim
    num_classes = int(train_dataset.classes)
    dim_num = int(train_dataset.data_x.shape[1])
    ts_len = int(train_dataset.data_x.shape[2])
    time_node_num = int(ts_len / args.seg_len)
    model = main_model_ood(args.seg_len, args.d_model1, args.d_model2, args.r1, args.r2, args.k_hops, dim_num,
                           time_node_num, args.pe_ratio,
                           num_classes)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
    label = torch.arange(start=0, end=dim_num, step=1)
    early_stopping = EarlyStopping(args.save_path, patience=args.patience)
    for epoch in tqdm(range(args.max_epoch)):
        setup_seed(args.seed)
        model.train()
        task_correct = 0
        tr_loss_task = 0
        domain_correct = 0
        tr_loss_domain = 0
        for i, (x, task_label) in enumerate(train_ts):
            setup_seed(args.seed)
            x = x.to(device)
            task_label = task_label.to(device)
            task_outputs, domain_output, _ = model(x)

            domain_label = label.repeat(1, task_label.shape[0]).T.to(device).squeeze(1)

            task_loss = loss_fn(task_outputs, task_label.long())
            domain_loss = loss_fn(domain_output, domain_label.long())
            loss = task_loss + domain_loss * args.alpha
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            task_pre = torch.argmax(task_outputs, dim=1)
            domain_pre = torch.argmax(domain_output, dim=1)
            task_correct += ((task_pre == task_label.long()).sum().item())
            domain_correct += ((domain_pre == domain_label.long()).sum().item())
            tr_loss_task += task_loss.item()
            tr_loss_domain += domain_loss.item()
        gc.collect()
        torch.cuda.empty_cache()
        train_acc = task_correct / train_data_num

        # --------------------------------val---------------------------------------
        val_correct = 0
        val_loss = 0
        model.eval()
        for i, (x, val_label) in enumerate(val_ts):
            x = x.to(device)
            val_label = val_label.to(device)
            val_outputs, domain_output, _ = model(x)
            val_loss += loss_fn(val_outputs, val_label.long()).item()
            val_correct += ((torch.argmax(val_outputs, dim=1) == val_label.long()).sum().item())
        val_acc = val_correct / val_data_num

        # print(
        #     "epoch: {}. train_acc:{:.6f}. train_loss:{:.6f} val_acc: {:.6f}. val_loss: {:.6f}".format(epoch, train_acc,
        #                                                                                               tr_loss_task / len(train_ts),
        #                                                                                               val_acc,
        #                                                                                               val_loss / len(val_ts)))
        early_stopping(val_loss / len(val_ts), model, args.data)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        #     ----------------------------------test------------------------------------------

    test_correct = 0
    model.load_state_dict(torch.load('{}/model_{}.pth'.format(args.save_path, args.data)))
    model.eval()
    with torch.no_grad():
        for i, (x, task_label) in enumerate(test_ts):
            x = x.to(device)
            task_label = task_label.to(device)
            task_outputs, domain_output, _ = model(x)

            task_pre = torch.argmax(task_outputs, dim=1)
            test_correct += ((task_pre == task_label.long()).sum().item())

            if i == 0:
                all_outputs = task_outputs
                all_label = task_label
            else:
                all_outputs = torch.cat((all_outputs, task_outputs), dim=0)
                all_label = torch.cat((all_label, task_label), dim=0)

    if num_classes == 2:
        onehot_label = label_binarize(all_label.cpu().detach().numpy(), classes=np.arange(num_classes))
        auprc = average_precision_score(onehot_label, F.softmax(all_outputs, dim=1)[:, 1].cpu().detach().numpy())
    else:
        onehot_label = label_binarize(all_label.cpu().detach().numpy(), classes=np.arange(num_classes))
        auprc = average_precision_score(onehot_label, F.softmax(all_outputs, dim=1).cpu().detach().numpy())

    test_acc = test_correct / test_data_num

    with open(args.result_path, 'a') as f:
        print("dataset: {} seed: {} acc:{:.6f}, auprc:{:.6f}".format(args.data, args.seed, test_acc, auprc),
              file=f)
    f.close()
