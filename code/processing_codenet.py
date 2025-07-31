import numpy as np
import pandas as pd
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as transforms
from config_codenet import Config

# Graph-based Knowledge Tracing: Modeling Student Proficiency Using Graph Neural Network.
# For more information, please refer to https://dl.acm.org/doi/10.1145/3350546.3352513
# Author: jhljx
# Email: jhljx8918@gmail.com

transform = transforms.Compose([
    transforms.ToTensor(),
])

class KTDataset(Dataset):
    def __init__(self, features, questions, answers):
        super(KTDataset, self).__init__()
        self.features = features
        self.questions = questions
        self.answers = answers

    def __getitem__(self, index):
        return self.features[index], self.questions[index], self.answers[index]

    def __len__(self):
        return len(self.features)


def pad_collate(batch):
    (features, questions, answers) = zip(*batch)
    features = [torch.FloatTensor(feat) for feat in features]
    questions = [torch.LongTensor(qt) for qt in questions]
    answers = [torch.FloatTensor(ans) for ans in answers]
    feature_pad = pad_sequence(features, batch_first=True, padding_value=-1.0)
    question_pad = pad_sequence(questions, batch_first=True, padding_value=-1.0)
    answer_pad = pad_sequence(answers, batch_first=True, padding_value=-1.0)
    return feature_pad, question_pad, answer_pad

def load_q_kc_matrix(file_path):
    """
    读取宽格式的题目-知识点矩阵
    :param file_path: CSV文件路径
    :return: (problem_ids, kc_ids, one_hot_matrix)
    """
    # 读取CSV，第一列作为索引（problem_id）
    df = pd.read_csv(file_path, index_col=0)
    
    # 提取有序的题目ID和知识点ID列表
    problem_ids = df.index.tolist()
    kc_ids = df.columns.tolist()
    
    # 转换为numpy矩阵
    one_hot_matrix = df.values.astype(np.float32)
    
    # 验证数据有效性
    assert len(problem_ids) == one_hot_matrix.shape[0]
    assert len(kc_ids) == one_hot_matrix.shape[1]
    print(f"矩阵加载成功 | 题目数: {len(problem_ids)} | 知识点数: {len(kc_ids)}")
    
    return kc_ids, one_hot_matrix


def preprocess_train_dataset(train_dataset, target_len=50, expand_factor=10, padding_value=-1.0):
    """
    对train_dataset进行增广采样，返回新的样本列表
    每个样本根据自身seq_len采样扩展为多个长度为target_len的子样本
    """
    new_samples = []
    for features, questions, answers in train_dataset:
        seq_len = len(features)
        n_samples = max(1, int(seq_len / target_len * expand_factor))
        if seq_len <= target_len:
            new_samples.append((features, questions, answers))
        else:
            interval = (seq_len - target_len) / n_samples
            for i in range(n_samples):
                start = int(i * interval)
                # 在当前区间内随机选一个起点
                max_start = int((i + 1) * interval)
                if max_start > seq_len - target_len:
                    max_start = seq_len - target_len
                if max_start > start:
                    start = np.random.randint(start, max_start + 1)
                end = start + target_len
                f = features[start:end]
                q = questions[start:end]
                a = answers[start:end]
                new_samples.append((f, q, a))
    return new_samples

def load_dkt_dataset(file_path,  batch_size,
                     train_ratio=0.7, val_ratio=0.2, shuffle=True):
    r"""
    Parameters:
        file_path: input file path of knowledge tracing data
        batch_size: the size of a student batch
        shuffle: whether to shuffle the dataset or not
        use_cuda: whether to use GPU to accelerate training speed
    Return:
        concept_num: the number of all concepts(or questions)
        graph: the static graph is graph type is in ['Dense', 'Transition', 'DKT'], otherwise graph is None
        train_data_loader: data loader of the training dataset
        valid_data_loader: data loader of the validation dataset
        test_data_loader: data loader of the test dataset
    NOTE: stole some code from https://github.com/lccasagrande/Deep-Knowledge-Tracing/blob/master/deepkt/data_util.py
    """
    config = Config()
    df = pd.read_csv(file_path).rename(columns={
        'user_id': 'SubjectID',
        'problem_id': 'ProblemID',
        'AST_score': 'Score',
        'status': 'status'
    })
    
    # 根据status设置Score
    print("\n=== 处理前的状态统计 ===")
    # print(df['status'].value_counts())
    
    # 实现更合理的分数处理逻辑 - 基于错误类型的分级评分
    def calculate_score_by_status(status):
        """
        基于提交状态计算分数，反映学生的编程能力和学习进度
        
        分数设计理念：
        1.0 - 完全正确
        0.8 - 逻辑正确，格式问题（接近掌握）
        0.6 - 算法正确，效率问题（基本掌握）
        0.5 - 算法正确，内存问题（基本掌握）
        0.3 - 程序逻辑有问题（部分理解）
        0.2 - 答案错误（初步尝试）
        0.1 - 语法错误（基础薄弱）
        0.0 - 系统错误或其他
        """
        status_score_map = {
            'Accepted': 1.0,                    # 完全正确
            'WA: Presentation Error': 0.8,      # 逻辑正确但输出格式问题
            'Time Limit Exceeded': 0.6,         # 算法正确但效率不够
            'Memory Limit Exceeded': 0.5,       # 算法正确但内存使用问题
            'Runtime Error': 0.3,               # 程序运行时错误（数组越界、空指针等）
            'Wrong Answer': 0.2,                # 答案错误
            'Compile Error': 0.1,               # 编译错误（语法问题）
            'Output Limit Exceeded': 0.1,       # 输出限制（通常是无限循环）
            'Query Limit Exceeded': 0.0,        # 查询限制
            'Internal error': 0.0,              # 系统内部错误
            'Judge System Error': 0.0           # 评判系统错误
        }
        
        return status_score_map.get(status, 0.0)

    # 应用新的分数计算逻辑
    # df['Score'] = df['status'].apply(calculate_score_by_status)

    # 不使用ast分数 根据status赋分 如果Accepted 则1.0 否则0.0
    # df['Score'] = df['status'].apply(lambda x: 1.0 if x == 'Accepted' else 0.0)
    
    # 添加分数统计信息
    # print("\n=== 新分数处理后的统计 ===")
    # score_stats = df['Score'].value_counts().sort_index(ascending=False)
    # print("分数分布:")
    # for score, count in score_stats.items():
    #     percentage = (count / len(df)) * 100
    #     print(f"  {score:.1f}: {count:6d} ({percentage:5.1f}%)")

    # print(f"\n平均分数: {df['Score'].mean():.3f}")
    # print(f"分数标准差: {df['Score'].std():.3f}")

    # 先将problem id转为整数
    # 去掉 'problem_id' 列中每个值的前缀 'p' 并转换为整数

    df['ProblemID'] = df['ProblemID'].str.lstrip('p').astype(int)
    print("原始数据集总数:", len(df))
    print("原始ProblemID的范围:", df['ProblemID'].min(), "->", df['ProblemID'].max())
    print("原始唯一ProblemID数量:", df['ProblemID'].nunique())
    print("原始学生的数量:", df['SubjectID'].nunique())

    # Step 1 - 基础过滤
    df.dropna(subset=['Score', 'ProblemID'], inplace=True)
    
    # 过滤少于1条记录的用户
    df = df.groupby('SubjectID').filter(lambda q: len(q) > 1).copy()

    # ===== 关键步骤 =====
    # 1.2 生成连续ID映射字典
    unique_ids = df['ProblemID'].sort_values().unique()  # 按原值排序
    id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}

    # 1.3 直接覆盖ProblemID列
    df['ProblemID'] = df['ProblemID'].map(id_mapping)

    # 在预处理完成后添加验证代码
    print("转换后ProblemID的范围:", df['ProblemID'].min(), "->", df['ProblemID'].max())
    print("唯一ProblemID数量:", df['ProblemID'].nunique())

    # 3. ===== 构建题目-知识点映射 =====
    # 构建题目-知识点映射关系（需要先加载矩阵）
    kc_ids, one_hot_matrix = load_q_kc_matrix(config.q_kc_path)
    total_concept_num = len(kc_ids)
    concept_threshold = int(total_concept_num * 0.7)  
    
    # 过滤没有知识点的题目
    df = df[df['ProblemID'].apply(lambda x: x < len(one_hot_matrix) and one_hot_matrix[x].sum() > 0)].copy()
    
    df['correct'] = df['Score']

    # 筛选出学生回答的问题涉及的知识点总数占总知识点的60%的学生
    def filter_by_concept_coverage(group):
        # 获取该学生回答的所有问题ID
        student_problems = group['ProblemID'].unique()
        
        # 统计涉及的知识点
        involved_concepts = set()
        for problem_id in student_problems:
            if problem_id < len(one_hot_matrix):  # 确保问题ID在矩阵范围内
                # 获取该问题涉及的知识点
                problem_concepts = np.where(one_hot_matrix[problem_id] == 1)[0]
                involved_concepts.update(problem_concepts)
        
        # 计算知识点覆盖度
        concept_coverage = len(involved_concepts)
        coverage_ratio = concept_coverage / total_concept_num
        
        # print(f"学生 {group['SubjectID'].iloc[0]}: 涉及 {concept_coverage}/{total_concept_num} 个知识点 ({coverage_ratio:.2%})")
        
        # 返回是否满足60%的条件
        return concept_coverage >= concept_threshold
    
    print("正在筛选学生（基于知识点覆盖度≥70%）...")
    df = df.groupby('SubjectID').filter(filter_by_concept_coverage).copy()
    print(f"经过知识点覆盖度筛选后的学生数量: {df['SubjectID'].nunique()}")

     # Step 4 - Convert to a sequence per user id and shift features 1 timestep
    feature_list = []
    question_list = []
    answer_list = []
    seq_len_list = []

    def get_data(series):
        # 每个学生每次提交的分数
        feature_list.append(series['Score'].astype(float).tolist())
        # 每个学生每次回答的问题id
        question_list.append(series['ProblemID'].tolist())
        # 每个学生每次回答的问题结果
        # answer_list.append((series['correct'] >= 0.9).astype('int').tolist())  # 修改这里
        # seq_len_list.append(series['correct'].shape[0])
        answer_list.append(series['correct'].eq(1).astype('int').tolist())
        seq_len_list.append(series['correct'].shape[0])


    df.groupby('SubjectID').apply(get_data)
    max_seq_len = np.max(seq_len_list)
    print('max seq_len: ', max_seq_len)
    student_num = len(seq_len_list)
    print('student num: ', student_num)
    feature_dim = int(df['Score'].max() + 1)
    print('feature_dim: ', feature_dim)
    # question_dim = len(one_hot_matrix.shape[0])
    # print('question_dim: ', question_dim)
    qt_num = one_hot_matrix.shape[0]
    concept_num = one_hot_matrix.shape[1]

    # 一切筛选完成后：统计正负样本的数量
    positive_samples = sum(row.count(1) for row in answer_list)
    negative_samples = sum(row.count(0) for row in answer_list)

    # 打印结果
    print(f"正样本数量：{positive_samples}")
    print(f"负样本数量：{negative_samples}")

    # 筛选学生后，计算题目难度
    # 只考虑筛选后数据集中存在的题目
    unique_problems = df['ProblemID'].unique()
    # print(f"筛选后数据集中的题目数量: {len(unique_problems)}")

   # 答题总数字典：记录每道题目的总提交次数
    total_submit = {pid: 0 for pid in unique_problems}
    # 答对题数字典：记录每道题目的正确提交次数
    right_submit = {pid: 0 for pid in unique_problems}
    
    # 遍历每个学生的答题记录
    for i in range(len(question_list)):
        qes_seq = question_list[i]
        ans_seq = feature_list[i]  # 使用特征列表中的二值化答题情况
        
        for j in range(len(qes_seq)):
            qid = qes_seq[j]
            # 只处理存在于筛选后数据集中的题目
            if qid in total_submit:
                total_submit[qid] += 1
                if ans_seq[j] == 1:  # 答对
                    right_submit[qid] += 1
  
    # 计算题目难度
    qt_difficult = {}
    for qid in unique_problems:
        total = total_submit[qid]
        right = right_submit[qid]
        avg_score = right / total if total > 0 else 0.0
        qt_difficult[qid] = 1.0 - avg_score
    
    # 创建全题目难度列表（0到矩阵大小-1），默认0.5
    qt_difficult_list = [0.5] * len(one_hot_matrix)
    # 更新有数据的题目
    for qid, diff in qt_difficult.items():
        if qid < len(qt_difficult_list):
            qt_difficult_list[qid] = diff
        else:
            print(f"警告: 题目ID {qid} 超出矩阵范围，忽略")
    
    
    # 打印前10题的难度值
    # for qid, diff in list(qt_difficult.items())[:10]:
    #     print(f"题目 {qid} | 总提交: {total_submit[qid]} | 正确次数: {right_submit.get(qid, 0)} | 难度: {diff:.2f}")

    # 在数据预处理流水线末端添加
    # print("== 数据验证 ==")
    # print(f"总题目数: {df['ProblemID'].nunique()}")
    # print(f"最大映射ID: {df['ProblemID'].max()}")
    # print(f"难度字典覆盖ID数: {len(qt_difficult_list)}")
    # print(f"是否所有ID都有难度值: {set(df['ProblemID'].unique()) == set(qt_difficult.keys())}")
     # ===== 4. 过滤后题目统计 =====
    print("\n=== 过滤后题目统计 ===")
    # 统计过滤后题目回答情况
    print(f"过滤后涉及题目数量: {len(unique_problems)}")
    print(f"过滤后题目数量占原始题目数的比例: {len(unique_problems) / len(one_hot_matrix)}")

    kt_dataset = KTDataset(feature_list, question_list, answer_list)
    train_size = int(train_ratio * student_num)
    val_size = int(val_ratio * student_num)
    test_size = student_num - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(kt_dataset, [train_size, val_size, test_size])
    print('train_size: ', train_size, 'val_size: ', val_size, 'test_size: ', test_size)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)
    valid_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)
    return qt_num, concept_num, train_data_loader, valid_data_loader, test_data_loader, one_hot_matrix, qt_difficult_list
