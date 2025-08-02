import numpy as np
import time
import random
import argparse
import pickle
import os
import gc
import datetime
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.cuda.amp import GradScaler
from torch.utils.data import  DataLoader
from metrics_codenet import KTLoss
from processing_codenet import load_dkt_dataset, KTDataset, pad_collate, preprocess_train_dataset
from config_codenet import Config
from early_stopping import EarlyStopping
from train_demo_1 import CEKT

# Graph-based Knowledge Tracing: Modeling Student Proficiency Using Graph Neural Network.
# For more information, please refer to https://dl.acm.org/doi/10.1145/3350546.3352513
# Author: jhljx
# Email: jhljx8918@gmail.com

config = Config()

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--data-dir', type=str, default='../data', help='Data dir for loading input data.')
# parser.add_argument('--data-file', type=str, default=config.dataPath, help='Name of input data file.')
parser.add_argument('--save-dir', type=str, default='../model', help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('-graph-save-dir', type=str, default='graphs', help='Dir for saving concept graphs.')
parser.add_argument('--load-dir', type=str, default='', help='Where to load the trained model if finetunning. ' + 'Leave empty to train from scratch')
parser.add_argument('--hid-dim', type=int, default=32, help='Dimension of hidden knowledge states.')
parser.add_argument('--emb-dim', type=int, default=32, help='Dimension of concept embedding.')
parser.add_argument('--attn-dim', type=int, default=32, help='Dimension of multi-head attention layers.')
parser.add_argument('--vae-encoder-dim', type=int, default=32, help='Dimension of hidden layers in vae encoder.')
parser.add_argument('--vae-decoder-dim', type=int, default=32, help='Dimension of hidden layers in vae decoder.')
parser.add_argument('--edge-types', type=int, default=2, help='The number of edge types to infer.')
parser.add_argument('--dropout', type=float, default=0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--bias', type=bool, default=True, help='Whether to add bias for neural network layers.')
parser.add_argument('--binary', type=bool, default=True, help='Whether only use 0/1 for results.')
parser.add_argument('--result-type', type=int, default=12, help='Number of results types when multiple results are used.')
parser.add_argument('--temp', type=float, default=0.5, help='Temperature for Gumbel softmax.')
parser.add_argument('--hard', action='store_true', default=False, help='Uses discrete samples in training forward pass.')
parser.add_argument('--no-factor', action='store_true', default=False, help='Disables factor graph model.')
parser.add_argument('--prior', action='store_true', default=False, help='Whether to use sparsity prior.')
parser.add_argument('--var', type=float, default=1, help='Output variance.')
parser.add_argument('--epochs', type=int, default=config.epoch, help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=config.batchSize, help='Number of samples per batch.')
parser.add_argument('--train-ratio', type=float, default=0.6, help='The ratio of training samples in a dataset.')
parser.add_argument('--val-ratio', type=float, default=0.2, help='The ratio of validation samples in a dataset.')
parser.add_argument('--shuffle', type=bool, default=True, help='Whether to shuffle the dataset or not.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--lr-decay', type=int, default=50, help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default=0.5, help='LR decay factor.')
parser.add_argument('--test', type=bool, default=False, help='Whether to test for existed model.')
parser.add_argument('--test-model-dir', type=str, default='logs/expDKT', help='Existed model file dir.')
parser.add_argument('--weight-method', type=str, default='senet',
                    choices=['senet', 'transformer'],
                    help='Method for calculating concept weights: senet or transformer (default: senet)')
parser.add_argument('--amp', action='store_true', default=False, help='Enable Automatic Mixed Precision training.')
parser.add_argument('--gpu-id', type=int, default=0, help='GPU device ID to use.')

# GPU信息显示
print("=== GPU信息 ===")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"可用GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        # 显示GPU内存信息
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  总内存: {gpu_memory:.1f} GB")
else:
    print("CUDA不可用，将使用CPU训练")
print("=" * 50)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.factor = not args.no_factor

# 设置GPU设备
if args.cuda:
    if args.gpu_id >= torch.cuda.device_count():
        print(f"警告：指定的GPU ID {args.gpu_id} 不存在，使用GPU 0")
        args.gpu_id = 0
    torch.cuda.set_device(args.gpu_id)
    print(f"使用GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")
else:
    print("使用CPU训练")

print(args)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # 针对RTX 4090优化
    torch.backends.cudnn.benchmark = True  # 改为True以提升性能
    torch.backends.cudnn.deterministic = False  # 改为False以提升性能
    print("已启用CUDNN优化，训练性能将提升")
res_len = 1 if args.binary else args.result_type

# Save model and meta-data. Always saves in a new sub-folder.
log = None
save_dir = args.save_dir
if args.save_dir:
    exp_counter = 0
    now = datetime.datetime.now()
    # timestamp = now.isoformat()
    timestamp = now.strftime('%Y-%m-%d %H-%M-%S')
    model_file_name = 'CEKT'
    save_dir = '{}/exp{}/'.format(args.save_dir, model_file_name + timestamp)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    meta_file = os.path.join(save_dir, 'metadata.pkl')
    model_file = os.path.join(save_dir, model_file_name + '.pt')
    optimizer_file = os.path.join(save_dir, model_file_name + '-Optimizer.pt')
    scheduler_file = os.path.join(save_dir, model_file_name + '-Scheduler.pt')
    log_file = os.path.join(save_dir, 'log.txt')
    log = open(log_file, 'w')
    pickle.dump({'args': args}, open(meta_file, "wb"))
else:
    print("WARNING: No save_dir provided!" + "Testing (within this script) will throw an error.")


# load dataset
saved_data_dir = os.path.join(args.data_dir, config.assignment)
print("从pkl文件加载数据集..."+saved_data_dir)
if os.path.exists(saved_data_dir):
    # 从pkl文件加载数据集
    print("从pkl文件加载数据集..."+saved_data_dir)
    
    with open(os.path.join(saved_data_dir, 'dataset_info.pkl'), 'rb') as f:
        dataset_info = pickle.load(f)
    
    qt_num = dataset_info['qt_num']
    concept_num = dataset_info['concept_num']
    qt_one_hot_matrix = dataset_info['one_hot_matrix']
    qt_difficult_list = dataset_info['qt_difficult_list']
    


    # 直接加载已划分的数据集
    def load_dataset_from_pkl(file_path):
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)
        return dataset
    
    train_dataset = load_dataset_from_pkl(os.path.join(saved_data_dir, 'train_dataset.pkl'))
    val_dataset = load_dataset_from_pkl(os.path.join(saved_data_dir, 'val_dataset.pkl'))
    test_dataset = load_dataset_from_pkl(os.path.join(saved_data_dir, 'test_dataset.pkl'))
    questions_embeddings = load_dataset_from_pkl(os.path.join(saved_data_dir, 'questions_embeddings_index_32.pkl'))
    # questions_id_2_index = load_dataset_from_pkl(os.path.join(saved_data_dir, 'new_original_id_2_index.pkl'))
    # kc_embeddings = load_dataset_from_pkl(os.path.join(saved_data_dir, 'kc_embeddings_768.pkl'))
    # kc_adj = np.load(os.path.join(saved_data_dir, 'adj_kc_codebert_codenet_Java.npy'),allow_pickle=True)

        # 将数据移到GPU（如果可用）
    if args.cuda:
        qt_one_hot_matrix = torch.tensor(qt_one_hot_matrix, dtype=torch.float32).cuda()
        # kc_adj = torch.tensor(kc_adj, dtype=torch.float32).cuda()
    else:
        qt_one_hot_matrix = torch.tensor(qt_one_hot_matrix, dtype=torch.float32)
        # kc_adj = torch.tensor(kc_adj, dtype=torch.float32)

    # 创建DataLoader

    # train_dataset = preprocess_train_dataset(train_dataset)
    # val_dataset = preprocess_train_dataset(val_dataset)
    # test_dataset = preprocess_train_dataset(test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                            shuffle=args.shuffle, collate_fn=pad_collate)
    valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                            shuffle=args.shuffle, collate_fn=pad_collate)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                           shuffle=args.shuffle, collate_fn=pad_collate)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")


model = CEKT(qt_one_hot_matrix,questions_embeddings,config.d_k, config.d_e, config.d_r, config.d_h, config.d_k_enhanced, config.gat_alpha, config.gat_nheads, config.gat_dropout)

# 将模型移到GPU
if args.cuda:
    model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

kt_loss = KTLoss()
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay, gamma=args.gamma)

print(f"模型初始化完成")


def train(epoch, best_val_loss):
    global optimizer, scheduler  # 移除未定义的scaler
    t = time.time()
    loss_train = []
    auc_train = []
    acc_train = []
    precision_train = []
    recall_train = []
    f1_train = []

    first_attempt_auc_train = []
    first_attempt_acc_train = []

    model.train()
    for batch_idx, (features, questions, answers) in enumerate(train_loader):
        optimizer.zero_grad()
        t1 = time.time()
        if args.cuda:
            features, questions, answers = features.cuda(non_blocking=True), questions.cuda(non_blocking=True), answers.cuda(non_blocking=True)
        predictions = model(features, questions)
        loss, auc, acc, precision, recall, f1 = kt_loss(predictions, answers)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # 记录训练指标

        if auc != -1 and acc != -1:
            auc_train.append(auc)
            acc_train.append(acc)
            precision_train.append(precision)
            recall_train.append(recall)
            f1_train.append(f1)

        # 计算第一次尝试的 AUC 和 ACC
        if batch_idx == 0:  # 假设第一个 batch 是第一次尝试
            first_attempt_auc_train.append(auc)
            first_attempt_acc_train.append(acc)

        loss_train.append(float(loss.cpu().detach().numpy()))
        

        print(f'batch idx: {batch_idx}, loss: {loss.item():.6f} auc: {auc:.4f}, acc: {acc:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}, time: {time.time() - t1:.3f}s')
        
        # 每10个batch打印一次
        # if batch_idx % 10 == 0:
        #     print(f'batch idx: {batch_idx}, loss kt: {loss_kt.item():.6f} auc: {auc:.4f}, acc: {acc:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}, time: {time.time() - t1:.3f}s')
        
        # 检查模型参数是否包含NaN
        has_nan = False
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(f"警告：参数 {name} 包含NaN，重新初始化模型")
                has_nan = True
                break
        
        if has_nan:
            # 重新初始化模型
            model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
            # 重新初始化优化器
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay, gamma=args.gamma)
            
        del loss
        if args.cuda:
            torch.cuda.empty_cache()

    # 打印训练期间第一次尝试的 AUC 和 ACC
    print('First attempt - auc_train: {:.10f}'.format(np.mean(first_attempt_auc_train)),
          'acc_train: {:.10f}'.format(np.mean(first_attempt_acc_train)))

    loss_val = []
    kt_val = []

    auc_val = []
    acc_val = []
    precision_val = []
    recall_val = []
    f1_val = []

    first_attempt_auc_val = []
    first_attempt_acc_val = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (features, questions, answers) in enumerate(valid_loader):
            if args.cuda:
                features, questions, answers = features.cuda(), questions.cuda(), answers.cuda()
            pred_res = model(features, questions)
            loss, auc, acc, precision, recall, f1 = kt_loss(pred_res, answers)
            loss_val.append(float(loss.cpu().detach().numpy()))
            if auc != -1 and acc != -1:
                auc_val.append(auc)
                acc_val.append(acc)
                precision_val.append(precision)
                recall_val.append(recall)
                f1_val.append(f1)
            # 计算第一次尝试的 AUC 和 ACC
            if batch_idx == 0:  # 假设第一个 batch 是第一次尝试
                first_attempt_auc_val.append(auc)
                first_attempt_acc_val.append(acc)

            del loss
            if args.cuda:
                torch.cuda.empty_cache()

    # 打印验证期间第一次尝试的 AUC 和 ACC
    print('First attempt - auc_val: {:.10f}'.format(np.mean(first_attempt_auc_val)),
          'acc_val: {:.10f}'.format(np.mean(first_attempt_acc_val)))

    print('Epoch: {:04d}'.format(epoch),
          'loss_train: {:.10f}'.format(np.mean(loss_train)),
          'auc_train: {:.10f}'.format(np.mean(auc_train)),
          'acc_train: {:.10f}'.format(np.mean(acc_train)),
          'precision_train: {:.10f}'.format(np.mean(precision_train)),
          'recall_train: {:.10f}'.format(np.mean(recall_train)),
          'f1_train: {:.10f}'.format(np.mean(f1_train)),
          'loss_val: {:.10f}'.format(np.mean(loss_val)),
          'auc_val: {:.10f}'.format(np.mean(auc_val)),
          'acc_val: {:.10f}'.format(np.mean(acc_val)),
          'precision_val: {:.10f}'.format(np.mean(precision_val)),
          'recall_val: {:.10f}'.format(np.mean(recall_val)),
          'f1_val: {:.10f}'.format(np.mean(f1_val)),
          'time: {:.4f}s'.format(time.time() - t))

    if args.save_dir and np.mean(loss_val) < best_val_loss:
    # if args.save_dir and np.mean(auc_val) > best_acc_loss:
        print('Best model so far, saving...')
        torch.save(model.state_dict(), model_file)
        torch.save(optimizer.state_dict(), optimizer_file)
        torch.save(scheduler.state_dict(), scheduler_file)

        print('Epoch: {:04d}'.format(epoch),
              'loss_train: {:.10f}'.format(np.mean(loss_train)),
              'auc_train: {:.10f}'.format(np.mean(auc_train)),
              'acc_train: {:.10f}'.format(np.mean(acc_train)),
              'precision_train: {:.10f}'.format(np.mean(precision_train)),
              'recall_train: {:.10f}'.format(np.mean(recall_train)),
              'f1_train: {:.10f}'.format(np.mean(f1_train)),
              'loss_val: {:.10f}'.format(np.mean(loss_val)),
              'auc_val: {:.10f}'.format(np.mean(auc_val)),
              'acc_val: {:.10f}'.format(np.mean(acc_val)),
              'precision_val: {:.10f}'.format(np.mean(precision_val)),
              'recall_val: {:.10f}'.format(np.mean(recall_val)),
              'f1_val: {:.10f}'.format(np.mean(f1_val)),
              'time: {:.4f}s'.format(time.time() - t), file=log)
        log.flush()
    
    # 返回验证指标字典，用于提前停止
    val_metrics = {
        'val_loss': np.mean(loss_val),
        'val_auc': np.mean(auc_val) if len(auc_val) > 0 else -1,
        'val_acc': np.mean(acc_val) if len(acc_val) > 0 else -1,
        'val_f1': np.mean(f1_val) if len(f1_val) > 0 else -1
    }
    
    res = np.mean(loss_val)
    del loss_train
    del auc_train
    del acc_train
    del precision_train
    del recall_train
    del f1_train
    del loss_val
    del auc_val
    del acc_val
    del precision_val
    del recall_val
    del f1_val
    gc.collect()
    if args.cuda:
        torch.cuda.empty_cache()
    return res, val_metrics


def test():
    loss_test = []
    auc_test = []
    acc_test = []
    precision_test = []
    recall_test = []
    f1_test =[]

    first_attempt_auc_test = []
    first_attempt_acc_test = []

    model.eval()
    model.load_state_dict(torch.load(model_file))
    with torch.no_grad():
        for batch_idx, (features, questions, answers) in enumerate(test_loader):
            if args.cuda:
                features, questions, answers = features.cuda(), questions.cuda(), answers.cuda()
            pred_res = model(features, questions)
            loss, auc, acc, precision, recall, f1 = kt_loss(pred_res, answers)
            loss_test.append(float(loss.cpu().detach().numpy()))
            if auc != -1 and acc != -1:
                auc_test.append(auc)
                acc_test.append(acc)
                precision_test.append(precision)
                recall_test.append(recall)
                f1_test.append(f1)
            del loss

            # 计算第一次尝试的 AUC 和 ACC
            if batch_idx == 0:  # 假设第一个 batch 是第一次尝试
                first_attempt_auc_test.append(auc)
                first_attempt_acc_test.append(acc)

    print('--------------------------------')
    print('--------Testing-----------------')
    print('--------------------------------')
    print('loss_test: {:.10f}'.format(np.mean(loss_test)),
          'auc_test: {:.10f}'.format(np.mean(auc_test)),
          'acc_test: {:.10f}'.format(np.mean(acc_test)),
          'precision_test: {:.10f}'.format(np.mean(precision_test)),
          'recall_test: {:.10f}'.format(np.mean(recall_test)),
          'f1_test: {:.10f}'.format(np.mean(f1_test)),
          'auc_max: {:.10f}'.format(np.max(auc_test)),
          'acc_max: {:.10f}'.format(np.max(acc_test)),
          'auc_std_test: {:.10f}'.format(np.std(auc_test)),
          'acc_std_test: {:.10f}'.format(np.std(acc_test)),
          'first_attempt_auc_test: {:.10f}'.format(np.mean(first_attempt_auc_test)),
          'first_attempt_acc_test: {:.10f}'.format(np.mean(first_attempt_acc_test)))
    if args.save_dir:
        print('--------------------------------', file=log)
        print('--------Testing-----------------', file=log)
        print('--------------------------------', file=log)

        print('loss_test: {:.10f}'.format(np.mean(loss_test)),
              'auc_test: {:.10f}'.format(np.mean(auc_test)),
              'acc_test: {:.10f}'.format(np.mean(acc_test)),
              'precision_test: {:.10f}'.format(np.mean(precision_test)),
              'recall_test: {:.10f}'.format(np.mean(recall_test)),
              'f1_test: {:.10f}'.format(np.mean(f1_test)),
              'auc_max: {:.10f}'.format(np.max(auc_test)),
              'acc_max: {:.10f}'.format(np.max(acc_test)),
              'auc_std_test: {:.10f}'.format(np.std(auc_test)),
              'acc_std_test: {:.10f}'.format(np.std(acc_test)),
              'first_attempt_auc_test: {:.10f}'.format(np.mean(first_attempt_auc_test)),
              'first_attempt_acc_test: {:.10f}'.format(np.mean(first_attempt_acc_test)),
              file=log)
        log.flush()

    del loss_test
    del auc_test
    del acc_test
    del precision_test
    del recall_test
    del f1_test
    gc.collect()
    if args.cuda:
        torch.cuda.empty_cache()


if args.test is False:
    # Train model
    print('start training!')
    t_total = time.time()
    best_val_loss = np.inf
    best_epoch = 0
    
    # 初始化提前停止机制
    # 可以选择单一指标监控或多指标监控
    
    # 方案1: 单一指标监控 - 监控验证损失
    early_stopping = EarlyStopping(
        patience=15,           # 15个epoch没有改善就停止
        min_delta=0.001,       # 最小改善阈值
        mode='min',            # 损失越小越好
        monitor='val_loss',    # 监控验证损失
        restore_best_weights=True,  # 恢复最佳权重
        verbose=True,          # 输出详细信息
        save_best_model=True,  # 保存最佳模型
        checkpoint_dir=f'{save_dir}/checkpoints' if args.save_dir else './checkpoints'
    )
   
    print(f"开始训练，最大epoch数: {args.epochs}")
    print("提前停止配置已启用，监控验证损失")
    
    for epoch in range(args.epochs):
        val_loss, val_metrics = train(epoch, best_val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
        
        # 提前停止检查
        # 为单一指标监控传递指标值
        should_stop = early_stopping(
            current_score=val_metrics['val_loss'],
            model=model,
            epoch=epoch,
            optimizer=optimizer,
            scheduler=scheduler,
            additional_info={
                'train_metrics': val_metrics,
                'current_lr': optimizer.param_groups[0]['lr']
            }
        )
        
  
        if should_stop:
            print(f"\n=== 提前停止训练 ===")
            print(f"停止epoch: {epoch}")
            print(f"最佳epoch: {early_stopping.get_best_epoch()}")
            print(f"最佳验证损失: {early_stopping.get_best_score():.6f}")
            print(f"总训练时间: {time.time() - t_total:.2f}秒")
            break
    else:
        print(f"\n=== 完成所有epoch训练 ===")
        print(f"最佳epoch: {best_epoch}")
        print(f"最佳验证损失: {best_val_loss:.6f}")
        print(f"总训练时间: {time.time() - t_total:.2f}秒")
    
    print("Optimization Finished!")
    print("Best Epoch: {:04d}".format(best_epoch))
    if args.save_dir:
        print("Best Epoch: {:04d}".format(best_epoch), file=log)
        log.flush()

test()
if log is not None:
    print(save_dir)
    log.close()