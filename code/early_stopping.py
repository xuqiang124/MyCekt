import numpy as np
import torch
import copy
import os
from typing import Optional, Union, Dict, Any
import logging

class EarlyStopping:
    """
    提前停止机制类，用于监控训练过程中的验证指标，避免过拟合和不必要的计算成本
    
    Args:
        patience (int): 连续多少个epoch没有改善就停止训练，默认10
        min_delta (float): 最小改善阈值，低于此值不算改善，默认0.0
        mode (str): 监控模式，'min'表示越小越好(如loss)，'max'表示越大越好(如accuracy, auc)
        monitor (str): 监控的指标名称，如'val_loss', 'val_auc', 'val_acc'等
        restore_best_weights (bool): 是否在停止时恢复最佳权重，默认True
        verbose (bool): 是否输出详细信息，默认True
        save_best_model (bool): 是否保存最佳模型，默认True
        checkpoint_dir (str): 检查点保存目录，默认'./checkpoints'
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        monitor: str = 'val_loss',
        restore_best_weights: bool = True,
        verbose: bool = True,
        save_best_model: bool = True,
        checkpoint_dir: str = './checkpoints'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode.lower()
        self.monitor = monitor
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self.save_best_model = save_best_model
        self.checkpoint_dir = checkpoint_dir
        
        # 验证mode参数
        if self.mode not in ['min', 'max']:
            raise ValueError(f"Mode {self.mode} is unknown, please choose from ['min', 'max']")
        
        # 创建检查点目录
        if self.save_best_model:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 初始化状态变量
        self.reset()
        
        # 设置比较函数
        if self.mode == 'min':
            self.monitor_op = np.less
            self.best_score = np.inf
        else:  # mode == 'max'
            self.monitor_op = np.greater
            self.best_score = -np.inf
        
        if self.verbose:
            print(f"EarlyStopping: 监控 {self.monitor} ({'最小值' if self.mode == 'min' else '最大值'})，"
                  f"耐心值 {self.patience}，最小变化阈值 {self.min_delta}")
    
    def reset(self):
        """重置提前停止状态"""
        self.wait = 0
        self.stopped_epoch = 0
        self.stop_training = False
        self.best_weights = None
        self.best_epoch = 0
        
    def __call__(
        self,
        current_score: float,
        model: torch.nn.Module,
        epoch: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        检查是否应该停止训练
        
        Args:
            current_score: 当前epoch的监控指标值
            model: 要监控的模型
            epoch: 当前epoch数
            optimizer: 优化器(可选)
            scheduler: 学习率调度器(可选)
            additional_info: 额外信息字典(可选)
        
        Returns:
            bool: 是否应该停止训练
        """
        
        # 检查是否有改善
        if self._is_improvement(current_score):
            self.best_score = current_score
            self.best_epoch = epoch
            self.wait = 0
            
            # 保存最佳权重
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())
            
            # 保存最佳模型到文件
            if self.save_best_model:
                self._save_checkpoint(model, optimizer, scheduler, epoch, current_score, additional_info)
            
            if self.verbose:
                print(f"Epoch {epoch:04d}: {self.monitor} 改善从 {self.best_score:.6f} 到 {current_score:.6f}")
        
        else:
            self.wait += 1
            if self.verbose:
                print(f"Epoch {epoch:04d}: {self.monitor} 没有改善 {current_score:.6f}，"
                      f"最佳值 {self.best_score:.6f}，等待 {self.wait}/{self.patience}")
            
            # 检查是否应该停止
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True
                
                if self.verbose:
                    print(f"\nEpoch {epoch:04d}: 提前停止")
                    print(f"最佳{self.monitor}: {self.best_score:.6f} (Epoch {self.best_epoch:04d})")
                
                # 恢复最佳权重
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    if self.verbose:
                        print(f"已恢复到最佳模型权重 (Epoch {self.best_epoch:04d})")
                
                return True
        
        return False
    
    def _is_improvement(self, current_score: float) -> bool:
        """检查当前分数是否比最佳分数有改善"""
        if self.mode == 'min':
            return current_score < (self.best_score - self.min_delta)
        else:  # mode == 'max'
            return current_score > (self.best_score + self.min_delta)
    
    def _save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any],
        epoch: int,
        score: float,
        additional_info: Optional[Dict[str, Any]] = None
    ):
        """保存检查点文件"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'score': score,
            'monitor': self.monitor,
            'mode': self.mode
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if additional_info is not None:
            checkpoint['additional_info'] = additional_info
        
        checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
        torch.save(checkpoint, checkpoint_path)
        
        if self.verbose:
            print(f"已保存最佳模型到 {checkpoint_path}")
    
    def load_best_model(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        加载最佳模型
        
        Returns:
            dict: 包含加载信息的字典
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"未找到检查点文件: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.verbose:
            print(f"已加载最佳模型 (Epoch {checkpoint['epoch']}, "
                  f"{checkpoint['monitor']}: {checkpoint['score']:.6f})")
        
        return {
            'epoch': checkpoint['epoch'],
            'score': checkpoint['score'],
            'additional_info': checkpoint.get('additional_info', {})
        }
    
    def get_best_score(self) -> float:
        """获取最佳分数"""
        return self.best_score
    
    def get_best_epoch(self) -> int:
        """获取最佳epoch"""
        return self.best_epoch
    
    def get_patience_info(self) -> Dict[str, int]:
        """获取耐心值相关信息"""
        return {
            'wait': self.wait,
            'patience': self.patience,
            'remaining_patience': self.patience - self.wait
        }


class MultiMetricEarlyStopping:
    """
    多指标提前停止机制，可以同时监控多个指标
    
    Args:
        early_stopping_configs (list): EarlyStopping配置列表
        combination_mode (str): 组合模式，'any'表示任意指标触发就停止，'all'表示所有指标都触发才停止
    """
    
    def __init__(
        self,
        early_stopping_configs: list,
        combination_mode: str = 'any'
    ):
        self.early_stoppers = []
        for config in early_stopping_configs:
            self.early_stoppers.append(EarlyStopping(**config))
        
        self.combination_mode = combination_mode.lower()
        if self.combination_mode not in ['any', 'all']:
            raise ValueError("combination_mode must be 'any' or 'all'")
    
    def __call__(
        self,
        metrics: Dict[str, float],
        model: torch.nn.Module,
        epoch: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        检查是否应该停止训练
        
        Args:
            metrics: 指标字典，如{'val_loss': 0.5, 'val_auc': 0.8}
            model: 模型
            epoch: 当前epoch
            optimizer: 优化器
            scheduler: 调度器
            additional_info: 额外信息
        
        Returns:
            bool: 是否应该停止训练
        """
        should_stop_list = []
        
        for early_stopper in self.early_stoppers:
            metric_name = early_stopper.monitor
            if metric_name in metrics:
                should_stop = early_stopper(
                    metrics[metric_name], model, epoch, optimizer, scheduler, additional_info
                )
                should_stop_list.append(should_stop)
            else:
                print(f"警告: 指标 '{metric_name}' 不在提供的metrics中")
                should_stop_list.append(False)
        
        if self.combination_mode == 'any':
            return any(should_stop_list)
        else:  # 'all'
            return all(should_stop_list)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取所有监控器的汇总信息"""
        summary = {}
        for i, early_stopper in enumerate(self.early_stoppers):
            summary[f'{early_stopper.monitor}'] = {
                'best_score': early_stopper.get_best_score(),
                'best_epoch': early_stopper.get_best_epoch(),
                'patience_info': early_stopper.get_patience_info()
            }
        return summary 