import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, f1_score, roc_curve, auc, \
    mean_squared_error, recall_score



class KTLoss(nn.Module):

    def __init__(self):
        super(KTLoss, self).__init__()

    def forward(self, pred_answers, real_answers):
        r"""
        Parameters:
            pred_answers: the correct probability of questions answered at the next timestamp
            real_answers: the real results(0 or 1) of questions answered at the next timestamp
        Shape:
            pred_answers: [batch_size, seq_len - 1]
            real_answers: [batch_size, seq_len]
        Return:
        """
        real_answers = real_answers[:, 1:]  # timestamp=1 ~ T
        # real_answers shape: [batch_size, seq_len - 1]
        # Here we can directly use nn.BCELoss, but this loss doesn't have ignore_index function
        answer_mask = torch.ne(real_answers, -1)
        # 更改预测值标签
        # pred_answers = torch.where(pred_answers >0.5, torch.ones_like(pred_answers), pred_answers)
        pred_one, pred_zero = pred_answers, 1.0 - pred_answers  # [batch_size, seq_len - 1]

        # calculate auc and accuracy metrics
        try:
            y_true = real_answers[answer_mask]
            y_pred = pred_one[answer_mask]

            # 对y_true进行筛选：如果值大于1则赋值为1否则为0
            y_true_binary = torch.where(y_true >= 1, 1.0, 0.0)
            
            print(f"y_pred: {y_pred}")
            # 处理NaN和无穷大值 - 使用torch方法
            y_pred = torch.where(torch.isnan(y_pred), torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype), y_pred)
            y_pred = torch.where(torch.isinf(y_pred) & (y_pred > 0), torch.tensor(1.0, device=y_pred.device, dtype=y_pred.dtype), y_pred)
            y_pred = torch.where(torch.isinf(y_pred) & (y_pred < 0), torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype), y_pred)

            criterion = nn.BCELoss()
            loss = criterion(y_pred, y_true_binary)

            y_true_binary = y_true_binary.cpu().detach().numpy()
            y_pred = y_pred.cpu().detach().numpy()
            

            # y_true_binary = y_true[y_true >= 0.5]
            y_pred_binary = np.where(y_pred >= 0.6, 1, 0)

            # print(y_true_binary)
            auc = roc_auc_score(y_true_binary, y_pred)
            # print(auc)
            acc = accuracy_score(y_true_binary, y_pred_binary)
            # print(acc)
            count(y_true_binary, y_pred_binary)
            
            precision = precision_score(y_true_binary, y_pred_binary,zero_division=0)
            # print("Precision Score:", precision)
            recall = recall_score(y_true_binary, y_pred_binary,zero_division=0)
            # print("Recall Score:", recall)
            # 计算F1分数
            f1 = f1_score(y_true_binary, y_pred_binary,zero_division=0)
            # print("F1 Score:", f1)
            # rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            # print("RMSE:", rmse)
        except ValueError as e:
            print(e)
            loss = torch.tensor(0.0, device=pred_answers.device, requires_grad=True)
            auc, acc, precision, recall, f1 = -1, -1, -1, -1, -1

        return loss, auc, acc, precision, recall, f1


def count(y_true_binary, y_pred):
    # 初始化 TP、TN、FP 和 FN 计数器
    tp = 0  # TP ：预测为正样本，实际也是正样本。
    tn = 0  # TN ：预测为负样本，实际也是负样本。
    fp = 0  # FP ：预测为正样本，实际是负样本。
    fn = 0  # FN ：预测为负样本，实际是正样本。
    #
    # # 假设 labels 是包含标签的列表
    neg_count = np.count_nonzero(y_true_binary == 0)  # 统计负样本的数量
    pos_count = np.count_nonzero(y_true_binary == 1)  # 统计正样本的数量
    # neg_count + pos_count == y_true.size()
    print("负样本数量：", neg_count)
    print("正样本数量：", pos_count)

    # pred_threshold = 1 - pos_count / neg_count
    # 预测标签中，大于等于threshold的置为1
    y_pred_binary = np.zeros_like(y_pred)
    y_pred_binary[y_pred >= 0.5] = 1  # 将 y_pred 大于等于 阈值(0.5) 的元素置为 1

    # 遍历每个样本的真实标签和预测结果
    for true_label, predicted_label in zip(y_true_binary, y_pred_binary):
        # 判断是否为 TP (正确判断正例)
        if true_label == 1 and predicted_label == 1:
            tp += 1
        # 判断是否为 TN (正确判断负例)
        elif true_label == 0 and predicted_label == 0:
            tn += 1
        # 判断是否为 FP (实际为负，错误判断为正)
        elif true_label == 0 and predicted_label == 1:
            fp += 1
        # 判断是否为 FN (实际为正，错误判断为负)
        elif true_label == 1 and predicted_label == 0:
            fn += 1

    # 打印结果
    print("TP: ", tp)
    print("TN: ", tn)
    print("FP: ", fp)
    print("FN: ", fn)



# 方法1计算auc
def calculate_auc_func1(y_labels, y_scores):
    pos_sample_ids = [i for i in range(len(y_labels)) if y_labels[i] == 1]
    neg_sample_ids = [i for i in range(len(y_labels)) if y_labels[i] == 0]

    sum_indicator_value = 0
    for i in pos_sample_ids:
        for j in neg_sample_ids:
            if y_scores[i] > y_scores[j]:
                sum_indicator_value += 1
            elif y_scores[i] == y_scores[j]:
                sum_indicator_value += 0.5

    # 避免除0错误
    if len(pos_sample_ids) > 0 and len(neg_sample_ids) > 0:
        auc = sum_indicator_value/(len(pos_sample_ids) * len(neg_sample_ids))
        print('AUC calculated by function1 is {:.2f}'.format(auc))
        return auc
    else:
        print('AUC cannot be calculated: no positive or negative samples')
        return 0.5

# 方法2计算auc, 当预测分相同时，未按照定义使用排序值的均值，而是直接使用排序值，当数据量大时，对auc影响小
def calculate_auc_func2(y_labels, y_scores):
    samples = list(zip(y_scores, y_labels))
    rank = [(values2, values1) for values1, values2 in sorted(samples, key=lambda x:x[0])]
    pos_rank = [i+1 for i in range(len(rank)) if rank[i][0] == 1]
    pos_cnt = np.sum(y_labels == 1)
    neg_cnt = np.sum(y_labels == 0)
    # 避免除0错误
    if pos_cnt > 0 and neg_cnt > 0:
        auc = (np.sum(pos_rank) - pos_cnt*(pos_cnt+1)/2) / (pos_cnt*neg_cnt)
        print('AUC calculated by function2 is {:.2f}'.format(auc))
        return auc
    else:
        print('AUC cannot be calculated: no positive or negative samples')
        return 0.5

