# coding: utf-8
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from config_codenet import Config
from layers import MLP, EraseAddGate, MLPEncoder, MLPDecoder, SELayer, TransformerWeightLayer
from utils import gumbel_softmax, get_kc_embedding, plot_concept_weights


# Graph-based Knowledge Tracing: Modeling Student Proficiency Using Graph Neural Network.
# For more information, please refer to https://dl.acm.org/doi/10.1145/3350546.3352513
# Author: jhljx
# Email: jhljx8918@gmail.com

config = Config()

class GKT(nn.Module):

    def __init__(self, qt_num, concept_num, hidden_dim, embedding_dim, edge_type_num, qt_one_hot_matrix,
                 qt_difficult_list, graph_model=None, weight_method='senet',
                 dropout=0.5, bias=True, binary=False, has_cuda=False):
        super(GKT, self).__init__()
        self.qt_num = qt_num
        self.concept_num = concept_num
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.edge_type_num = edge_type_num
        self.res_len = 2 if binary else 12
        self.has_cuda = has_cuda
        self.qt_kc_one_hot = qt_one_hot_matrix
        # TODO 2023/12/6 加一个属性kc_embedding
        # TODO 2023/12/6 加一个可训练参数 neigh_weight
        self.neigh_weight = nn.Parameter(torch.tensor(0.8))
        if self.has_cuda:
            self.qt_kc_one_hot = torch.from_numpy(self.qt_kc_one_hot).cuda()
            self.kc_embeddings = torch.load(config.embedding_path).cuda() # 加载.pt文件
            # self.kc_embedding = get_kc_embedding().cuda()
            # self.question_embedding = get_question_embedding().cuda()
            self.qt_diffcult = torch.tensor(qt_difficult_list).cuda()
        else:
            self.qt_kc_one_hot = torch.from_numpy(self.qt_kc_one_hot)
            # self.kc_embedding = get_kc_embedding()
            self.kc_embeddings = torch.load(config.embedding_path, 'cpu') # 加载.pt文件
            # self.question_embedding = get_question_embedding()
            self.qt_diffcult = torch.tensor(qt_difficult_list)
        zero_padding = torch.zeros(1, self.concept_num, device=self.qt_kc_one_hot.device)
        self.qt_kc_one_hot = torch.cat((self.qt_kc_one_hot, zero_padding), dim=0)
        self.graph_model = graph_model
        self.weight_method = weight_method
        # one-hot feature and question
        one_hot_feat = torch.eye(self.concept_num)
        self.one_hot_feat = one_hot_feat.cuda() if self.has_cuda else one_hot_feat
        ### 4/19 改动
        # self.se_layer = SELayer(channel=self.concept_num, reduction=4)
        ### 4/19 改动
        if self.weight_method == 'senet':
            self.se_layer = SELayer(channel=self.concept_num, reduction=4)  # 原SENet实现
        elif self.weight_method == 'transformer':
            self.weight_layer = TransformerWeightLayer(
                feature_dim=self.embedding_dim,
                nhead=4,  # 可添加为命令行参数
                num_layers=2  # 可添加为命令行参数
            )
        # concept and concept & response embeddings
        self.emb_x = nn.Embedding(self.res_len * concept_num, embedding_dim)
        # last embedding is used for padding, so dim + 1
        self.emb_c = nn.Embedding(concept_num + 1, embedding_dim, padding_idx=-1)
        self.softmax = nn.Softmax(dim=1)  

        # f_self function and f_neighbor functions
        mlp_input_dim = hidden_dim + embedding_dim
        self.f_self = MLP(mlp_input_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias)
        self.f_neighbor_list = nn.ModuleList()

        for i in range(edge_type_num):
            self.f_neighbor_list.append(MLP(2 * mlp_input_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias))

        # Erase & Add Gate
        self.erase_add_gate = EraseAddGate(hidden_dim, concept_num)
        # Gate Recurrent Unit
        self.gru = nn.GRUCell(hidden_dim, hidden_dim, bias=bias)
        # prediction layer
        self.predict = nn.Linear(hidden_dim, 1, bias=bias)
    
    
    # Aggregate step, as shown in Section 3.2.1 of the paper
    def _aggregate(self, xt, qt, ht):
        r"""
        Parameters:
            xt: input one-hot question answering features at the current timestamp
            qt: question indices for all students in a batch at the current timestamp
            ht: hidden representations of all concepts at the current timestamp
            batch_size: the size of a student batch
        Shape:
            xt: [batch_size]
            qt: [batch_size]
            ht: [batch_size, concept_num, hidden_dim]
            tmp_ht: [batch_size, concept_num, hidden_dim + embedding_dim]
        Return:
            tmp_ht: aggregation results of concept hidden knowledge state and concept(& response) embedding
        """
        # print("\n=== 进入_aggregate函数 ===")
        # 将没有回答的置为false
        qt_mask = torch.ne(qt, -1)  # [mask_num, ]
        qt = torch.where(qt != -1, qt, self.qt_num * torch.ones_like(qt, device=qt.device)) # [mask_num, ] 如果为-1，置为qt_num
        # 问题与知识点的one-hot对应
        masked_feat = F.embedding(qt[qt_mask], self.qt_kc_one_hot.long())  # [mask_num, concept_num]

        # 知识点文本向量 kc_text_vector
        x_embedding = self.kc_embeddings # [concept_num, emb_dim]
        # 每个题目难度的系数 qt_d
        qt_d = F.embedding(qt[qt_mask], self.qt_diffcult.unsqueeze(1)) # [mask_num, 1]
        # 检查点1：检查qt_d的计算
        # print("\n=== 检查点1: qt_d计算 ===")
        # print(f"qt_d范围: [{qt_d.min()}, {qt_d.max()}]")
        # print(f"qt_d是否有NaN: {torch.isnan(qt_d).any()}")
        
        # 计算知识点嵌入
        qt_multi_KC_vector = masked_feat.unsqueeze(2) * x_embedding # [mask_num, concept_num, emb_dim]
        
        # 检查expanded_x_embedding
        expanded_x_embedding = x_embedding.unsqueeze(0).repeat(qt.shape[0], 1, 1)
        
        # 检查q_kc_embedding计算过程
        new_q_kc_emb = expanded_x_embedding.clone()
        # print("\n=== 知识点嵌入计算过程 ===")
        # print(f"masked_feat中1的数量: {masked_feat.sum()}")
        # print(f"每个样本涉及的知识点数: {masked_feat.sum(dim=1)}")
        
        for i in range(masked_feat.shape[0]):
            active_kcs = (masked_feat[i] == 1).nonzero(as_tuple=True)[0]
            if len(active_kcs) == 0:
                print(f"警告: 样本 {i} 没有激活的知识点")
                # 如果存在未激活的知识点 跳过
        
        # qt_multi_KC_vector经过se层，得到每个问题 知识点 的 权重
        #TODO 202504 知识点权重计算分支 -- 存在问题 nan
        if self.weight_method == 'senet':
            weights = self.se_layer(qt_multi_KC_vector)
            weights = self.softmax(weights)  # [mask_num, concept_num] 经过归一化后的注意力分数
        elif self.weight_method == 'transformer':
            weights = self.weight_layer(qt_multi_KC_vector)
        # 先将 masked_feat 与 qt_d相乘，得到   每个涉及知识点的位置上都乘上该题目的难度
        tmp_d = masked_feat * qt_d  # [mask_num, concept_num]
        # 检查点2：检查tmp_d的计算
        # print("\n=== 检查点2: tmp_d计算 ===")
        # print(f"tmp_d范围: [{tmp_d.min()}, {tmp_d.max()}]")
        # print(f"tmp_d是否有NaN: {torch.isnan(tmp_d).any()}")
        
        # xt为自身特征（分数），扩大量纲 * 1000
        xt = xt[qt_mask] * 1000  # [mask_num, ]
         # 检查点3：检查xt的计算
        # print("\n=== 检查点3: xt计算 ===")
        # print(f"xt范围: [{xt.min()}, {xt.max()}]")
        # print(f"xt是否有NaN: {torch.isnan(xt).any()}")
        # 调整形状，方便后面相乘
        xt_ = xt.reshape(-1, 1) # [mask_num, 1]
        # tmp_d = masked_feat * qt_d，即masked_feat * qt_d * xt_
        temp = tmp_d * xt_  # [mask_num, concept_num]
        # 检查点4：检查temp的计算
        # print("\n=== 检查点4: temp计算 ===")
        # print(f"temp范围: [{temp.min()}, {temp.max()}]")
        # print(f"temp是否有NaN: {torch.isnan(temp).any()}")
        # masked_feat[one hot] * qt_d[难度系数] * xt_[ast分数] * se_feat_[知识点权重] ， 得到融合的特征se_fex
        # se_fex = temp * weights  #题-知识点o [mask_num, concept_num]
        # TODO 2025/05/17
        se_fex = temp * weights
        # 问题涉及的知识点的部分，se_fex[i][j] * kc_embedding[j],
        # 将[concept_num, emb_dim] 扩张到 [batch_size, concept_num, embedding_dim]
        expanded_x_embedding = x_embedding.unsqueeze(0).repeat(qt.shape[0], 1, 1) # [batch_size, concept_num, embedding_dim]
        # 从qt_kc_onehot中，取出qt对应的行
        # concept_idx_mat = F.embedding(qt, self.qt_kc_one_hot.long())  # [batch_size, concept_num]
        # 经过emb_c层，即嵌入层
        # q_kc_embedding = self.emb_c(concept_idx_mat)  # [batch_size, concept_num, embedding_dim]
        q_kc_embedding = expanded_x_embedding # qc_vector表示知识点的嵌入
        # TODO: 更新qc_vector对应位置（回答的问题涉及的知识点）上的值为
        #  x_embedding[i][j] * 知识点权重se_feat[i][j] * 问题难度qt_d[i] * ast分数xt[i]
        new_q_kc_emb = q_kc_embedding.clone()
        
        for i in range(masked_feat.shape[0]):
            for j in range(masked_feat.shape[1]):
                if masked_feat[i, j] == 1: # 即是问题对应的知识点
                    kc_emb = expanded_x_embedding[i, j]
                    t = se_fex[i, j]
                    fusion_emb = kc_emb * t
                   

        q_kc_embedding = new_q_kc_emb  # 赋值给 q_kc_embedding
        # TODO 改动 -----------------
        # ht初始是全0的三维张量【初始每个学生关于知识点的隐藏状态】+ res_embedding【输入一个答题序列后，更新的知识点自身特征向量+自身权重特征】
        # tmp_ht 将ht和qc_vector(知识点嵌入)在最后一个维度拼接起来，用于后续的更新
        tmp_ht = torch.cat((ht, q_kc_embedding), dim=-1)  # [batch_size, concept_num, hidden_dim + embedding_dim]
        # 检查是否有NaN
        if torch.isnan(tmp_ht).any():
            print("tmp_ht中NaN的位置:")
            nan_indices = torch.where(torch.isnan(tmp_ht))
            print(f"NaN索引: {nan_indices}")
        return tmp_ht

    # GNN aggregation step, as shown in 3.3.2 Equation 1 of the paper
    def _agg_neighbors(self, tmp_ht, qt):
        r"""
        Parameters:
            tmp_ht: temporal hidden representations of all concepts after the aggregate step
            - 在聚合过程之后得到的所有知识点的临时隐藏状态（包含隐藏状态ht和嵌入向量concept_embedding）
            qt: question indices for all students in a batch at the current timestamp
            - 当前时间片下学生回答问题的编号
        Shape:
            tmp_ht: [batch_size, concept_num, hidden_dim + embedding_dim]
            qt: [batch_size]
            m_next: [batch_size, concept_num, hidden_dim]
        Return:
            m_next: hidden representations of all concepts aggregating neighboring representations at the next timestamp
            concept_embedding: input of VAE (optional)
            rec_embedding: reconstructed input of VAE (optional)
            z_prob: probability distribution of latent variable z in VAE (optional)
        """
        # 将没有回答的置为false
        qt_mask = torch.ne(qt, -1)  # [batch_size, ], qt != -1
        masked_qt = qt[qt_mask]  # [mask_num, ]
        # 从隐藏状态中取出回答了的问题对应的知识点的隐藏状态
        masked_tmp_ht = tmp_ht[qt_mask]  # [mask_num, concept_num, hidden_dim + embedding_dim]
        # TODO 2024/4/6 改动
        masked_feat = F.embedding(qt[qt_mask], self.qt_kc_one_hot.long()).unsqueeze(2)  # [mask_num, concept_num, 1]，问题与知识点对应的one hot，最后增加一个维度，方便与masked_tmp_ht相乘

        # # 从masked_tmp_ht取出自身(回答了的知识点的)隐藏状态
        self_ht_ = masked_tmp_ht * masked_feat
        self_feat = self.f_self(self_ht_)  # [mask_num, concept_num, hidden_dim] # 问题，知识点及其自身隐藏状态
        
        # 检查self_feat是否有NaN并进行修复
        if torch.isnan(self_feat).any():
            self_feat = torch.zeros_like(self_feat)
        
        # 限制self_feat的范围
        self_feat = torch.clamp(self_feat, min=-10.0, max=10.0)
        
        # 拼接自身状态和隐藏状态
        # 其中，self_ht_表示当前时间步中的知识点的特征表示，masked_tmp_ht是经过掩码的、包含有历史信息的知识点的特征表示。
        #  neigh_ht是当前时间步中各个知识点的相邻节点的特征表示向量。
        neigh_ht = torch.cat((self_ht_, masked_tmp_ht), dim=-1) #  [mask_num, concept_num, 2 * (hidden_dim + embedding_dim)]

        # 检查neigh_ht是否有NaN，这是所有后续计算的基础
        if torch.isnan(neigh_ht).any():
            neigh_ht = torch.zeros_like(neigh_ht)
        
        # 限制neigh_ht的范围
        neigh_ht = torch.clamp(neigh_ht, min=-5.0, max=5.0)
        
        # TODO 2023/12/5 换成kc_embedding,通过知识点文本嵌入，计算KC之间相似度——>构建知识点图
        concept_embedding = self.kc_embeddings
        sp_send, sp_rec, sp_send_t, sp_rec_t = self._get_edges(masked_qt)
        graphs, rec_embedding, z_prob = self.graph_model(concept_embedding, sp_send, sp_rec, sp_send_t, sp_rec_t) # VAE构图
        
        # 检查VAE输出的graphs是否有NaN并进行修复
        for k in range(graphs.shape[0]):
            if torch.isnan(graphs[k]).any():
                graphs[k] = torch.where(torch.isnan(graphs[k]), 
                                       torch.zeros_like(graphs[k]), 
                                       graphs[k])
        
        neigh_features = 0
        for k in range(self.edge_type_num):
            # 拿到masked_qt中每个qt对应的kc_id
            mask_qt_kc = self.qt_kc_one_hot[masked_qt] # [mask_num, concept_num]
            # 问题对应的知识点的id
            kc_id = torch.arange(self.concept_num).cuda() * mask_qt_kc if self.has_cuda else torch.arange(
                self.concept_num) * mask_qt_kc
            
            # 去掉没有涉及的知识点，形成一个list,包含问题及其涉及的知识点的id
            kc_id_non_zero = [row[row.nonzero(as_tuple=True)].long() for row in kc_id]
            
            # 取出对应kc_id的加和平均
            sublist = kc_id_non_zero[0]
            
            # 在计算adj前检查输入
            if len(sublist) > 0:
                graph_slice_check = graphs[k][sublist, :]
                
                if torch.isnan(graph_slice_check).any():
                    graph_slice_check = torch.where(torch.isnan(graph_slice_check), 
                                                   torch.zeros_like(graph_slice_check), 
                                                   graph_slice_check)
                
                # adj为知识点权重的平均值
                adj = torch.mean(graph_slice_check.unsqueeze(dim=-1), dim=0).unsqueeze(dim=0)
            else:
                adj = torch.zeros((1, self.concept_num, 1), device=graphs.device)
            
            # 限制adj的范围
            adj = torch.clamp(adj, min=-5.0, max=5.0)
            
            # 计算邻居特征时添加数值保护
            neighbor_output = self.f_neighbor_list[k](neigh_ht)
            
            # 检查MLP输出是否有NaN
            if torch.isnan(neighbor_output).any():
                neighbor_output = torch.zeros_like(neighbor_output)
            
            # 限制MLP输出范围
            neighbor_output = torch.clamp(neighbor_output, min=-5.0, max=5.0)
            
            # 限制neigh_weight参数的范围防止其变得异常
            safe_neigh_weight = torch.clamp(self.neigh_weight, min=0.1, max=0.9)
            
            if k == 0: # 如果k = 0 ,
                neigh_features = adj * neighbor_output
            else: # k = 1
                # 使用安全的权重值，并在计算前检查所有输入
                if torch.isnan(neigh_features).any():
                    neigh_features = torch.zeros_like(neigh_features)
                    
                term1 = safe_neigh_weight * neigh_features
                term2 = (1 - safe_neigh_weight) * adj * neighbor_output
                
                # 检查每一项是否有NaN
                if torch.isnan(term1).any():
                    term1 = torch.zeros_like(term1)
                if torch.isnan(term2).any():
                    term2 = torch.zeros_like(term2)
                    
                neigh_features = term1 + term2
            
            # 每次计算后检查neigh_features是否有NaN
            if torch.isnan(neigh_features).any():
                neigh_features = torch.zeros_like(neigh_features)
            
            # 限制neigh_features范围
            neigh_features = torch.clamp(neigh_features, min=-5.0, max=5.0)
        
        # neigh_features: [mask_num, concept_num, hidden_dim]
        m_next = tmp_ht[:, :, :self.hidden_dim]  # [mask_num, concept_num, hidden_dim]
        # m_next 更新为 聚合后邻居的特征
        m_next[qt_mask] = neigh_features  # [mask_num, concept_num, hidden_dim]
        # TODO 2024/4/7 改动 在mask_feat为1的地方 更新为self_feature对应的值。
        # 遍历 masked_feat 中值为 1 的位置，将 m_next 中对应位置的值用 self_feat 替换 [涉及的知识点还是用自身特征]
        for i in range(masked_feat.shape[0]):
            for j in range(masked_feat.shape[1]):
                if masked_feat[i, j, 0] == 1:
                    m_next[i, j] = self_feat[i, j]

        # 限制m_next的范围防止数值溢出
        m_next = torch.clamp(m_next, min=-50.0, max=50.0)
        
        return m_next, concept_embedding, rec_embedding, z_prob

    # Update step, as shown in Section 3.3.2 of the paper
    def _update(self, tmp_ht, ht, qt):
        r"""
        Parameters:
            tmp_ht: temporal hidden representations of all concepts after the aggregate step
            ht: hidden representations of all concepts at the current timestamp
            qt: question indices for all students in a batch at the current timestamp
        Shape:
            tmp_ht: [batch_size, concept_num, hidden_dim + embedding_dim]
            ht: [batch_size, concept_num, hidden_dim]
            qt: [batch_size]
            h_next: [batch_size, concept_num, hidden_dim]
        Return:
            h_next: hidden representations of all concepts at the next timestamp
            concept_embedding: input of VAE (optional)
            rec_embedding: reconstructed input of VAE (optional)
            z_prob: probability distribution of latent variable z in VAE (optional)
        """
        # print("\n=== 进入_update函数 ===")
        # print("输入检查:")
        # print(f"tmp_ht范围: [{tmp_ht.min()}, {tmp_ht.max()}]")
        # print(f"tmp_ht是否有NaN: {torch.isnan(tmp_ht).any()}")
        # print(f"ht范围: [{ht.min()}, {ht.max()}]")
        # print(f"ht是否有NaN: {torch.isnan(ht).any()}")
        qt_mask = torch.ne(qt, -1)  # [batch_size], qt != -1
        mask_indices = torch.nonzero(qt_mask, as_tuple=True)
        mask_num = mask_indices[0].shape[0]

        # GNN Aggregation
        m_next, concept_embedding, rec_embedding, z_prob = self._agg_neighbors(tmp_ht,
                                                                               qt)  # [batch_size, concept_num, hidden_dim]
        
        # 限制m_next的范围防止数值溢出
        m_next = torch.clamp(m_next, min=-50.0, max=50.0)
        
        # Erase & Add Gate 通过擦除和添加门控制对应位置的信息更新
        m_next[qt_mask] = self.erase_add_gate(m_next[qt_mask])  # [mask_num, concept_num, hidden_dim]
        
        # 再次限制范围
        m_next = torch.clamp(m_next, min=-50.0, max=50.0)
        
        # GRU
        # m_next: 从邻居节点聚合后得到的节点信息
        # h_next: 表示隐藏状态，也可以理解为当前时刻的隐藏层状态。在循环神经网络（如GRU）中，h_next 可能用来存储模型在当前时间步的隐藏状态。
        # 之后 h_next 会被更新，并且传递给下一时刻
        h_next = m_next #下一步的隐藏状态，将使用 m_next 更新后的节点表示。
        '''
        在这里，m_next[qt_mask] 和 ht[qt_mask] 表示满足条件的邻居节点特征和当前节点特征。这些特征是按照批次和概念编号排列的，
        因此通过将其 reshape 成 (mask_num * concept_num, hidden_dim) 的形状，
        可以将多个批次的邻居节点和当前节点特征合并为一个大的批次，以便进行并行的 GRU 更新操作。
        '''
        # 通过 GRU 模型计算下一个时刻的节点表示，m_next[qt_mask] 和 ht[qt_mask] 是输入的两个张量，分别表示当前节点和隐藏状态。
        # reshape(-1, self.hidden_dim) 的目的是调整张量的形状以匹配 GRU 模型的输入要求。
        res = self.gru(m_next[qt_mask].reshape(-1, self.hidden_dim),
                       ht[qt_mask].reshape(-1, self.hidden_dim))  # [mask_num * concept_num, hidden_num]
        index_tuple = (torch.arange(mask_num, device=qt_mask.device),)  # 创建了一个由0到mask_num-1的索引组成的张量（tensor），这里索引的数量与掩码数量相同，用于指定要更新的位置。
        h_next[qt_mask] = h_next[qt_mask].index_put(index_tuple, res.reshape(-1, self.concept_num, self.hidden_dim)) # 将新的节点表示 res 更新到 h_next[qt_mask] 对应的位置上。
        return h_next, concept_embedding, rec_embedding, z_prob

    # Predict step, as shown in Section 3.3.3 of the paper
    def _predict(self, h_next, qt):
        r"""
        Parameters:
            h_next: hidden representations of all concepts at the next timestamp after the update step
            qt: question indices for all students in a batch at the current timestamp
        Shape:
            h_next: [batch_size, concept_num, hidden_dim]
            qt: [batch_size]
            y: [batch_size, concept_num]
        Return:
            y: predicted correct probability of all concepts at the next timestamp 下一个时间步，针对每个概念，学生回答相关问题的正确概率。
            有效问题位置（`qt_mask`为True）通过sigmoid转换为概率，无效位置可能保留原始值或未被使用。
        """
        qt_mask = torch.ne(qt, -1)  # [batch_size], qt != -1
        y = self.predict(h_next).squeeze(dim=-1)  # [batch_size, concept_num]
        y[qt_mask] = torch.sigmoid(y[qt_mask])  # [batch_size, concept_num]
        # print(y)
        return y

    def _get_next_pred(self, yt, next_qt):
        r"""
        Parameters:
            yt: predicted correct probability of all concepts at the next timestamp
            q_next: question index matrix at the next timestamp
            batch_size: the size of a student batch
        Shape:
            y: [batch_size, concept_num]
            questions: [batch_size, seq_len]
            pred: [batch_size, ]
        Return:
            pred: predicted correct probability of the question answered at the next timestamp
        """
        next_qt = torch.where(next_qt != -1, next_qt, self.qt_num * torch.ones_like(next_qt, device=yt.device))
        one_hot_qt = F.embedding(next_qt.long(), self.qt_kc_one_hot.long())  # 获取问题对应的知识点 有的问题没有知识点对应怎么办？
        # dot product between yt and one_hot_qt
        _pred = yt * one_hot_qt
        # TODO 2024/3/14 改动：对于每个样本，取所有非0的平均(涉及的所有知识点)
        non_zero_pred = []
        # # Iterate over each row and filter out values greater than 0
        for i in range(_pred.size(0)):
            row = _pred[i]
            non_zero_pred.append(row[row > 0])
        mean_values = [tensor.mean() for tensor in non_zero_pred]  # 所有非0的平均(涉及的所有知识点)
        # Convert the list of mean values to a tensor
        pred = torch.stack(mean_values) # [batch_size]

        return pred
    
    # Get edges for edge inference in VAE
    def _get_edges(self, masked_qt):
        r"""
        Parameters:
            masked_qt: qt index with -1 padding values removed
        Shape:
            masked_qt: [mask_num, ]
            rel_send: [edge_num, concept_num]
            rel_rec: [edge_num, concept_num]
        Return:
            rel_send: from nodes in edges which send messages to other nodes 发送消息至其他节点的边中的节点
            rel_rec:  to nodes in edges which receive messages from other nodes 从其他节点接收消息的边中的节点
        # """
        mask_qt_kc = self.qt_kc_one_hot[masked_qt] # 问题与知识点的对应
        
        kc_id = torch.arange(self.concept_num).cuda() if self.has_cuda else torch.arange(self.concept_num)
        mask_qt_kc_score = mask_qt_kc * kc_id # 问题对应的知识点id
        masked_qt = mask_qt_kc_score[mask_qt_kc_score != 0] # 筛选出不为0的（涉及的知识点） TODO 这里有点问题？
        mask_num = torch.count_nonzero(masked_qt).item()

        row_arr = masked_qt.cpu().numpy().reshape(-1, 1)  # [mask_num, 1]
        row_arr = np.repeat(row_arr, self.concept_num, axis=1)  # [mask_num, concept_num]
        col_arr = np.arange(self.concept_num).reshape(1, -1)  # [1, concept_num] 逆向边
        col_arr = np.repeat(col_arr, mask_num, axis=0)  # [mask_num, concept_num]
        # add reversed edges
        new_row = np.vstack((row_arr, col_arr))  # [2 * mask_num, concept_num]
        new_col = np.vstack((col_arr, row_arr))  # [2 * mask_num, concept_num]
        row_arr = new_row.flatten()  # [2 * mask_num * concept_num, ]
        col_arr = new_col.flatten()  # [2 * mask_num * concept_num, ]
        data_arr = np.ones(2 * mask_num * self.concept_num)
        
        # data_arr：用于填充稀疏矩阵的数据数组。
        # (row_arr, col_arr)：这是元组，表示了非零元素在稀疏矩阵中的位置。row_arr 包含行索引，col_arr 包含列索引，这两个数组的长度应该相同。
        # shape=(self.concept_num, self.concept_num)：指定了稀疏矩阵的形状，即行数和列数。
        init_graph = sp.coo_matrix((data_arr, (row_arr, col_arr)), shape=(self.concept_num, self.concept_num))
        init_graph.setdiag(0)  # remove self-loop edges 移除自环
        row_arr, col_arr, _ = sp.find(init_graph) # 使用 sp.find 函数从 init_graph 稀疏矩阵中提取非零元素的行索引和列索引，并忽略值。
        
        row_tensor = torch.from_numpy(row_arr).long()
        col_tensor = torch.from_numpy(col_arr).long()
        one_hot_table = torch.eye(self.concept_num, self.concept_num)
        rel_send = F.embedding(row_tensor, one_hot_table)  # [edge_num, concept_num]
        rel_rec = F.embedding(col_tensor, one_hot_table)  # [edge_num, concept_num]
        
        sp_rec, sp_send = rel_rec.to_sparse(), rel_send.to_sparse()
        sp_rec_t, sp_send_t = rel_rec.T.to_sparse(), rel_send.T.to_sparse()
        sp_send = sp_send.to(device=masked_qt.device)
        sp_rec = sp_rec.to(device=masked_qt.device)
        sp_send_t = sp_send_t.to(device=masked_qt.device)
        sp_rec_t = sp_rec_t.to(device=masked_qt.device)
        return sp_send, sp_rec, sp_send_t, sp_rec_t

    def forward(self, features, questions):
        r"""
        Parameters:
            features: input one-hot matrix
            questions: question index matrix
        seq_len dimension needs padding, because different students may have learning sequences with different lengths.
        Shape:
            features: [batch_size, seq_len]
            questions: [batch_size, seq_len]
            pred_res: [batch_size, seq_len - 1]
        Return:
            pred_res: the correct probability of questions answered at the next timestamp
            concept_embedding: input of VAE (optional)
            rec_embedding: reconstructed input of VAE (optional)
            z_prob: probability distribution of latent variable z in VAE (optional)
        """
        # 确保neigh_weight参数在合理范围内
        with torch.no_grad():
            if torch.isnan(self.neigh_weight).any() or torch.isinf(self.neigh_weight).any():
                self.neigh_weight.data.fill_(0.5)  # 重置为中性值
            else:
                self.neigh_weight.data.clamp_(0.1, 0.9)  # 限制在合理范围内
        
        batch_size, seq_len = features.shape
        
        ht = Variable(torch.zeros((batch_size, self.concept_num, self.hidden_dim), device=features.device))
        
        pred_list = []
        ec_list = []  # concept embedding list in VAE
        rec_list = []  # reconstructed embedding list in VAE
        z_prob_list = []  # probability distribution of latent variable z in VAE

        for i in range(seq_len):
            xt = features[:, i]
            qt = questions[:, i]
            qt_mask = torch.ne(qt, -1)
            
            tmp_ht = self._aggregate(xt, qt, ht)
            h_next, concept_embedding, rec_embedding, z_prob = self._update(tmp_ht, ht, qt) 
            
            ht[qt_mask] = h_next[qt_mask]  # update new ht
            
            if torch.isnan(ht).any():
                print("\n!!! 警告：检测到ht中有NaN，打印详细信息 !!!")
                print(f"h_next范围: [{h_next.min()}, {h_next.max()}]")
                print(f"qt_mask中True的数量: {qt_mask.sum()}")
                print(f"h_next[qt_mask]范围: [{h_next[qt_mask].min()}, {h_next[qt_mask].max()}]")
                break

            yt = self._predict(h_next, qt)  # [batch_size, concept_num]
            if i < seq_len - 1:
                pred = self._get_next_pred(yt, questions[:, i + 1])
                pred_list.append(pred)
            ec_list.append(concept_embedding)
            rec_list.append(rec_embedding)
            z_prob_list.append(z_prob)

           
        if len(pred_list) != 0:
            pred_res = torch.stack(pred_list, dim=1)  # [batch_size, seq_len - 1] 将pred_list堆叠
        else:
            # 返回一个空的 Tensor，形状可以是 [batch_size, 0]，具体形状根据后续代码需求确定
            pred_res = torch.empty(features.size(0), 0)
        return pred_res, ec_list, rec_list, z_prob_list


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, msg_hidden_dim, msg_output_dim, concept_num, edge_type_num=2,
                 tau=0.1, factor=True, dropout=0., bias=True):
        super(VAE, self).__init__()
        self.edge_type_num = edge_type_num
        self.concept_num = concept_num
        self.tau = tau
        self.encoder = MLPEncoder(input_dim, hidden_dim, output_dim, factor=factor, dropout=dropout, bias=bias)
        self.decoder = MLPDecoder(input_dim, msg_hidden_dim, msg_output_dim, hidden_dim, edge_type_num, dropout=dropout,
                                  bias=bias)
        # inferred latent graph, used for saving and visualization
        self.graphs = nn.Parameter(torch.zeros(edge_type_num, concept_num, concept_num))
        self.graphs.requires_grad = False

    def _get_graph(self, edges, sp_rec, sp_send):
        r"""
        Parameters:
            edges: sampled latent graph edge weights from the probability distribution of the latent variable z
            sp_rec: one-hot encoded receive-node index(sparse tensor)
            sp_send: one-hot encoded send-node index(sparse tensor)
        Shape:
            edges: [edge_num, edge_type_num]
            sp_rec: [edge_num, concept_num]
            sp_send: [edge_num, concept_num]
        Return:
            graphs: latent graph list modeled by z which has different edge types
        """
        # 检查是否有边数据
        if edges.shape[0] == 0 or sp_send.shape[0] == 0:
            graphs = Variable(torch.zeros(self.edge_type_num, self.concept_num, self.concept_num, device=edges.device))
            return graphs
        
        x_index = sp_send._indices()[1].long()  # send node index: [edge_num, ]
        y_index = sp_rec._indices()[1].long()  # receive node index [edge_num, ]
        
        graphs = Variable(torch.zeros(self.edge_type_num, self.concept_num, self.concept_num, device=edges.device))
        
        for k in range(self.edge_type_num):
            if len(x_index) > 0 and len(y_index) > 0:
                index_tuple = (x_index, y_index)
                graphs[k] = graphs[k].index_put(index_tuple, edges[:, k])  # used for calculation
                #############################
                # here, we need to detach edges when storing it into self.graphs in case memory leak!
                self.graphs.data[k] = self.graphs.data[k].index_put(index_tuple, edges[:,
                                                                                 k].detach())  # used for saving and visualization
                #############################
        
        return graphs

    def forward(self, data, sp_send, sp_rec, sp_send_t, sp_rec_t):
        r"""
              Parameters:
                  data: input concept embedding matrix
                  sp_send: one-hot encoded send-node index(sparse tensor)
                  sp_rec: one-hot encoded receive-node index(sparse tensor)
                  sp_send_t: one-hot encoded send-node index(sparse tensor, transpose)
                  sp_rec_t: one-hot encoded receive-node index(sparse tensor, transpose)
              Shape:
                  data: [concept_num, embedding_dim]
                  sp_send: [edge_num, concept_num]
                  sp_rec: [edge_num, concept_num]
                  sp_send_t: [concept_num, edge_num]
                  sp_rec_t: [concept_num, edge_num]
              Return:
                  graphs: latent graph list modeled by z which has different edge types
                  output: the reconstructed data
                  prob: q(z|x) distribution
              """
        logits = self.encoder(data, sp_send, sp_rec, sp_send_t, sp_rec_t)  # [edge_num, output_dim(edge_type_num)]
        
        # 如果logits为空或包含NaN，创建默认值
        if logits.numel() == 0 or torch.isnan(logits).any():
            logits = torch.zeros((1, self.edge_type_num), device=data.device)
        
        edges = gumbel_softmax(logits, tau=self.tau, dim=-1)  # [edge_num, edge_type_num]
        
        prob = F.softmax(logits, dim=-1)
        output = self.decoder(data, edges, sp_send, sp_rec, sp_send_t, sp_rec_t)  # [concept_num, embedding_dim]
        
        graphs = self._get_graph(edges, sp_send, sp_rec)
        
        return graphs, output, prob

