import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        
        # 有待更新，是否采用三个不同的邻接矩阵进行筛选
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh): # 先将所有的Wh复制N份，然后拼接起来，得到(N, N, 2 * out_features)
        N = Wh.size()[0]
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0) 
        Wh_repeated_alternating = Wh.repeat(N, 1) 
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1) 
        return all_combinations_matrix.view(N, N, 2 * self.out_features)  

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GATLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GATLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj) # 在论文中，GAT后直接是增强的KC表示，ELU可以在KAG内或外部处理
        return x

class KAG(nn.Module):
    def __init__(self, d_k_enhanced, d_e, d_r, d_h):
        super(KAG, self).__init__()
        # 按照论文的描述，输入维度可能是 GAT输出的维度 + 练习维度 + 回答维度 + 上一时刻隐藏状态维度
        input_dim = d_k_enhanced + d_e + d_r + d_h
        
        # 修正了论文中的命名和代码中的逻辑
        # W_g 对应论文中的 W_q (用于update gate g_t)
        # W_h_candidate 对应论文中的 W_h (用于candidate hidden state h_tilde)
        self.W_g = nn.Linear(input_dim, d_h) 
        self.W_h_candidate = nn.Linear(input_dim, d_h)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, h_kc_enhanced, e_t, r_t, prev_h):
        # 修复输入拼接逻辑，原始的拼接有问题
        combined_input = torch.cat((h_kc_enhanced, e_t, r_t, prev_h), dim=1)
        
        # 对应论文公式 (12) 和 (13)
        g_t = self.sigmoid(self.W_g(combined_input))
        h_candidate = self.tanh(self.W_h_candidate(combined_input))
        h_t = (1 - g_t) * prev_h + g_t * h_candidate
        return h_t

class OutputLayer(nn.Module):
    def __init__(self, d_h, d_e, d_k):
        super(OutputLayer, self).__init__()
        # 输入维度是 隐藏状态维度 + 下一题练习维度 + 下一题KC维度
        self.W_p = nn.Linear(d_h + d_e + d_k, 1) # 输出是一个概率值，所以维度是1
        self.sigmoid = nn.Sigmoid()

    def forward(self, h_t, e_next, kc_next):
        # 对应论文公式 (14)
        combined = torch.cat((h_t, e_next, kc_next), dim=1)
        y_pred = self.sigmoid(self.W_p(combined))
        return y_pred
    
def convert_embeddings_to_tensor(embeddings):
    """
    将embeddings列表转换为PyTorch张量
    
    Args:
        embeddings: 包含numpy数组的列表
    
    Returns:
        torch.Tensor: 浮点数张量
    """
    # 优化：先转换为numpy数组，再转换为tensor
    if isinstance(embeddings, list):
        embeddings_array = np.array(embeddings, dtype=np.float32)
        embeddings_tensor = torch.tensor(embeddings_array, dtype=torch.float32)
    else:
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    return embeddings_tensor

def convert_embeddings_to_Long_tensor(embeddings):
    """
    将embeddings列表转换为PyTorch Long类型张量
    
    Args:
        embeddings: 包含数据的列表或numpy数组
    
    Returns:
        torch.LongTensor: Long类型的张量
    """
    # 转换为Long类型的tensor
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.long)  
    return embeddings_tensor

class CEKT(nn.Module):
    def __init__(self, qt_one_hot_matrix, questions_embeddings, d_k, d_e, d_r, d_h, d_k_enhanced, alpha, nheads, dropout):
        super(CEKT, self).__init__()
        self.hidden_dim = d_h
        self.num_exercises = qt_one_hot_matrix.shape[0] 
        self.num_kcs = qt_one_hot_matrix.shape[1]
        # print(convert_embeddings_to_tensor(questions_embeddings).shape)

        # KC嵌入投影层
        # self.kc_proj = nn.Linear(768, d_k)
        # kc_projected = self.kc_proj(convert_embeddings_to_tensor(kc_embeddings))
        # self.kc_embedding = nn.Parameter(kc_projected)

        # 练习嵌入投影层
        # self.exercise_proj = nn.Linear(768, d_e)
        # # questions_embeddings是浮点数数据，应该使用convert_embeddings_to_tensor
        # projected = self.exercise_proj(convert_embeddings_to_tensor(questions_embeddings))  # (num_exercises, d_e)
        self.exercise_embedding = nn.Parameter(convert_embeddings_to_tensor(questions_embeddings), requires_grad=False) # 将练习从768维度减小为32维度, nums * 768 -> nums * 32，设置为静态不随梯度改变

        self.response_embedding = nn.Embedding(2, d_r) # 0 for wrong, 1 for correct
        # 确保qt_one_hot_matrix是tensor类型
        self.qt_one_hot_matrix = convert_embeddings_to_Long_tensor(qt_one_hot_matrix).detach()

        # KC-练习融合投影层（如果维度不匹配）
        if d_k != d_e:
            self.exercise_to_kc_proj = nn.Linear(d_e, d_k)
        else:
            self.exercise_to_kc_proj = None
            
        # GAT的输入是d_k, 输出是d_k_enhanced
        self.gat = GAT(nfeat=d_k, nhid=d_h, nclass=d_k_enhanced, dropout=dropout, alpha=alpha, nheads=nheads)
        self.kag = KAG(d_k_enhanced, d_e, d_r, d_h)
        self.output_layer = OutputLayer(d_h, d_e, d_k) # 输出层的KC输入是原始KC嵌入



    def forward(self, features, questions, adj):
        '''
        Args:
            features: (batch_size, seq_len) 学生答题的正确性，0表示错误，1表示正确
            questions: (batch_size, seq_len) 题目ID序列
            adj: 邻接矩阵，用于GAT的知识图谱
            
        Returns:
            predictions: (batch_size, seq_len-1) 对下一题的预测概率
        '''
        # 初始化
        batch_size, seq_len = features.shape
        
        # 确保设备一致性
        device = features.device
        prev_h = torch.zeros(batch_size, self.hidden_dim, device=device)
        
        # 将所有self变量移动到当前设备
        self.to(device)
        
        predictions = []
        
        # 模拟时序处理
        for t in range(seq_len - 1):
            # --- 1. 获取当前时间步 t 的数据 ---
            exercise_id_t = questions[:, t]  # shape: (batch_size,)
            response_t = features[:, t]      # shape: (batch_size,)

            # 直接通过exercise_id_t获取对应的KC one-hot向量
            # exercise_id_t: (batch_size,) - 题目ID
            # qt_one_hot_matrix: (num_exercises, num_kcs) - 题目到KC的映射矩阵
            # qt_one_hot_matrix已经在GPU上，直接使用
            kc_ids_t = self.qt_one_hot_matrix[exercise_id_t]  # (batch_size, num_kcs)
            
            # --- 2. 嵌入---
            # 使用矩阵乘法处理整个batch
            # kc_t_embedded = torch.mm(kc_ids_t, self.kc_embedding)  # (batch_size, d_k)
            e_t_embedded = self.exercise_embedding[exercise_id_t]  # (batch_size, d_e)

            # 根据answer_t生成response embedding
            # 如果answer_t为0，使用response_embedding[0]；否则使用response_embedding[1]
            response_indices = (response_t >= 0.5).long()  # 将answer_t转换为0或1的索引
            r_t_embedded = self.response_embedding(response_indices)     # (batch_size, d_r)


            # 更新原始知识点嵌入矩阵 - 融合练习信息和KC信息
            # 策略：使用kc_ids_t与e_t_embedded相乘，得到KC-练习融合信息

            # 计算KC-练习融合信息

            # 处理维度匹配问题
            if self.exercise_to_kc_proj is not None:
                # 需要投影层来匹配维度
                projected_exercise = self.exercise_to_kc_proj(e_t_embedded)  # (batch_size, d_k)
                enhanced_kc_embedded = projected_exercise  # (batch_size, d_k)
            else:
                # 维度匹配，直接叠加
                enhanced_kc_embedded = e_t_embedded  # (batch_size, d_k)
            
            # 将enhanced_kc_embedded广播为(batch_size, num_kcs, d_k)
            # 然后与kc_ids_t相乘，得到仅与当前题目相关的知识点信息矩阵
            enhanced_kc_broadcast = enhanced_kc_embedded.unsqueeze(1).expand(-1, self.num_kcs, -1)  # (batch_size, num_kcs, d_k)
            kc_ids_t_expanded = kc_ids_t.unsqueeze(-1).expand(-1, -1, enhanced_kc_embedded.size(-1))  # (batch_size, num_kcs, d_k)
            
            # 相乘得到与当前题目相关的知识点信息矩阵
            # 相关KC行有数值，不相关的KC行为0
            # 这样确保了只有与当前题目相关的KC才会被更新
            relevant_kc_info = enhanced_kc_broadcast * kc_ids_t_expanded  # (batch_size, num_kcs, d_k)
            
            # 更新KC嵌入矩阵 - 压缩batch维度并保留知识点关联信息
            # 核心思想：
            # 1. 压缩batch维度：聚合所有样本对同一KC的贡献
            # 2. 保留关联信息：利用KC间的相似度进行知识传播
            
            # 步骤1：压缩batch维度，得到每个KC的聚合信息
            # 对每个KC，计算所有batch样本中该KC的平均信息
            kc_aggregated_info = torch.sum(relevant_kc_info, dim=0)  # (num_kcs, d_k)
            kc_involvement_count = torch.sum(kc_ids_t, dim=0)  # (num_kcs,) - 每个KC被多少样本涉及
            
            
            # 步骤2：计算平均信息（避免被0除）
            # 使用更安全的方式避免梯度图断裂
            kc_involvement_count_expanded = kc_involvement_count.unsqueeze(1).expand_as(kc_aggregated_info)
            kc_avg_info = torch.where(
                kc_involvement_count_expanded > 0,
                kc_aggregated_info / (kc_involvement_count_expanded + 1e-8),
                torch.zeros_like(kc_aggregated_info)
            )

            # 步骤3：考虑KC间的关联信息
            # 使用KC间的相似度来增强更新
            # kc_similarity_matrix = torch.mm(kc_avg_info, kc_avg_info.t())  # (num_kcs, num_kcs)
            # kc_similarity_matrix = F.softmax(kc_similarity_matrix / 0.1, dim=1)  # 温度参数0.1
                      
            # --- 3. 知识增强 (GAT) ---
            # 真实场景中，GAT的输入是与 e_t 相关的KCs的嵌入
            # 这里简化：GAT在所有KC嵌入上操作，得到增强后的全局KC嵌入
            enhanced_all_kc_embeddings = self.gat(kc_avg_info, adj)
            
            # 提取当前KC的增强后表示
            # 使用矩阵乘法处理整个batch
            h_kc_enhanced_t = torch.mm(kc_ids_t.float(), enhanced_all_kc_embeddings) # (batch_size, d_k_enhanced)

            # --- 4. 状态更新 (KAG) ---
            h_t = self.kag(h_kc_enhanced_t, e_t_embedded, r_t_embedded, prev_h)
            
            # --- 5. 准备下一时间步 t+1 的数据用于预测 ---
            exercise_id_next = questions[:, t+1]  # shape: (batch_size,)
            
            # 直接通过exercise_id_next获取对应的KC one-hot向量
            kc_ids_next = self.qt_one_hot_matrix[exercise_id_next]  # (batch_size, num_kcs)
            
            e_next_embedded = self.exercise_embedding[exercise_id_next]  # (batch_size, d_e)
            # 使用kc_ids_next与kc_avg_info相乘来获取KC嵌入
            kc_next_embedded = torch.mm(kc_ids_next.float(), enhanced_all_kc_embeddings)  # (batch_size, d_k)
            
            # --- 6. 预测 (OutputLayer) ---
            y_pred = self.output_layer(h_t, e_next_embedded, kc_next_embedded)
            predictions.append(y_pred)
            
            # 更新 prev_h
            prev_h = h_t
        
        # 返回预测结果，确保维度正确
        # 将predictions列表转换为tensor，每个元素是(batch_size, 1)
        predictions_tensor = torch.stack(predictions, dim=1)  # (batch_size, seq_len-1, 1)
        predictions_tensor = predictions_tensor.squeeze(-1)  # (batch_size, seq_len-1
        return predictions_tensor  # (batch_size, seq_len-1)

