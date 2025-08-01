class Config:
    def __init__(self):
        # 训练参数
        self.learning_rate = 0.01
        self.epoch = 50
        self.assignment = 'Python'
        self.batchSize = 512
        self.shuffle = True

       
        # 维度定义
        self.d_k = 32           # 原始KC嵌入维度
        self.d_e = 32           # 练习嵌入维度
        self.d_r = 32           # 回答嵌入维度
        self.d_h = 32           # 隐藏状态/GAT中间层维度
        self.d_k_enhanced = 32  # GAT输出的增强KC维度

         # GAT参数
        self.gat_alpha = 0.2
        self.gat_nheads = 3
        self.gat_dropout = 0.1



