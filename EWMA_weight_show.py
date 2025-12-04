'''
本脚本用于计算并可视化不同lambda值下EWMA模型的权重分布
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from typing import List, Tuple

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class EWMAWeightVisualizer:
    """
    EWMA权重可视化工具类
    用于展示不同lambda值下EWMA模型对历史数据的权重分布
    """
    
    def __init__(self):
        pass
    
    def calculate_ewma_weights(self, n_days: int, lambda_: float) -> np.ndarray:
        """
        计算EWMA权重
        
        参数:
        n_days: 历史数据天数
        lambda_: 衰减因子
        
        返回:
        weights: 每一天的权重数组，从最新到最旧
        """
        # 创建时间索引，从0到n_days-1（0是最新的，n_days-1是最旧的）
        t = np.arange(n_days)
        
        # EWMA权重公式: w_t = (1-λ) * λ^t
        # 其中t=0对应最新数据，t越大对应越旧的数据
        weights = (1 - lambda_) * (lambda_ ** t)
        
        # 归一化权重（理论上应该接近1，但由于截断效应可能略小于1）
        weights = weights / np.sum(weights)
        
        return weights
    
    def plot_single_lambda_weights(self, n_days: int, lambda_: float, 
                                 ax=None, label: str = None) -> None:
        """
        绘制单个lambda值的权重分布图
        
        参数:
        n_days: 历史数据天数
        lambda_: 衰减因子
        ax: matplotlib轴对象，如果为None则创建新图
        label: 图例标签
        """
        weights = self.calculate_ewma_weights(n_days, lambda_)
        
        # 创建负数时间轴（从-1天到-n天）
        time_axis = -np.arange(1, n_days + 1)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        
        # 绘制权重曲线
        if label is None:
            label = f'λ = {lambda_}'
        
        ax.plot(time_axis, weights, marker='o', markersize=3, 
                linewidth=2, label=label, alpha=0.8)
        
        ax.set_xlabel('时间 (天)', fontsize=12)
        ax.set_ylabel('权重', fontsize=12)
        ax.set_title(f'EWMA权重分布 (λ = {lambda_}, N = {n_days}天)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 设置x轴刻度（负数）
        if n_days <= 50:
            ax.set_xticks(-np.arange(1, n_days + 1, 5))
        else:
            ax.set_xticks(-np.arange(1, n_days + 1, 10))
        
        # 确保图形在第二象限（x轴负数，y轴正数）
        ax.set_xlim(-n_days - 1, 0)
        ax.set_ylim(0, max(weights) * 1.1)
    
    def plot_multiple_lambda_weights(self, n_days: int, 
                                   lambda_values: List[float],
                                   figsize: Tuple[int, int] = (14, 10)) -> None:
        """
        绘制多个lambda值的权重分布对比图
        
        参数:
        n_days: 历史数据天数
        lambda_values: lambda值列表
        figsize: 图形大小
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # 颜色列表
        colors = plt.cm.Set1(np.linspace(0, 1, len(lambda_values)))
        
        # 第一个子图：线性坐标
        for i, lambda_ in enumerate(lambda_values):
            weights = self.calculate_ewma_weights(n_days, lambda_)
            time_axis = -np.arange(1, n_days + 1)  # 负数时间轴
            
            ax1.plot(time_axis, weights, marker='o', markersize=2, 
                    linewidth=2, label=f'λ = {lambda_}', 
                    color=colors[i], alpha=0.8)
        
        ax1.set_xlabel('时间 (天)', fontsize=12)
        ax1.set_ylabel('权重', fontsize=12)
        ax1.set_title(f'EWMA权重分布对比 (N = {n_days}天) - 线性坐标', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xlim(-n_days - 1, 0)  # 确保在第二象限
        
        # 第二个子图：对数坐标（更好地显示衰减特性）
        for i, lambda_ in enumerate(lambda_values):
            weights = self.calculate_ewma_weights(n_days, lambda_)
            time_axis = -np.arange(1, n_days + 1)  # 负数时间轴
            
            ax2.semilogy(time_axis, weights, marker='o', markersize=2, 
                        linewidth=2, label=f'λ = {lambda_}', 
                        color=colors[i], alpha=0.8)
        
        ax2.set_xlabel('时间 (天)', fontsize=12)
        ax2.set_ylabel('权重 (对数坐标)', fontsize=12)
        ax2.set_title(f'EWMA权重分布对比 (N = {n_days}天) - 对数坐标', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_xlim(-n_days - 1, 0)  # 确保在第二象限
        
        plt.tight_layout()
        plt.show()
    
    def analyze_weight_characteristics(self, n_days: int, 
                                     lambda_values: List[float]) -> pd.DataFrame:
        """
        分析不同lambda值下权重的特征
        
        参数:
        n_days: 历史数据天数
        lambda_values: lambda值列表
        
        返回:
        DataFrame: 包含各种权重特征的分析结果
        """
        results = []
        
        for lambda_ in lambda_values:
            weights = self.calculate_ewma_weights(n_days, lambda_)
            
            # 计算特征指标
            max_weight = np.max(weights)
            weight_50_pct = np.sum(weights[:int(n_days * 0.5)])  # 前50%时间的权重和
            weight_90_pct = np.sum(weights[:int(n_days * 0.9)])  # 前90%时间的权重和
            
            # 计算有效观测数（权重衰减到最大权重的1%时的天数）
            threshold = max_weight * 0.01
            effective_days = np.sum(weights >= threshold)
            
            # 计算权重的集中度（基尼系数的简化版本）
            sorted_weights = np.sort(weights)[::-1]  # 降序排列
            cumsum_weights = np.cumsum(sorted_weights)
            concentration = np.sum((2 * np.arange(1, len(weights) + 1) - len(weights) - 1) * sorted_weights) / (len(weights) * np.sum(weights))
            
            results.append({
                'Lambda': lambda_,
                '最大权重': max_weight,
                '前50%时间权重和': weight_50_pct,
                '前90%时间权重和': weight_90_pct,
                '有效观测天数': effective_days,
                '权重集中度': concentration
            })
        
        return pd.DataFrame(results)
    
    def plot_weight_heatmap(self, n_days: int, 
                          lambda_values: List[float],
                          figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        绘制权重热力图
        
        参数:
        n_days: 历史数据天数
        lambda_values: lambda值列表
        figsize: 图形大小
        """
        # 创建权重矩阵
        weight_matrix = np.zeros((len(lambda_values), n_days))
        
        for i, lambda_ in enumerate(lambda_values):
            weights = self.calculate_ewma_weights(n_days, lambda_)
            weight_matrix[i, :] = weights
        
        # 绘制热力图
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(weight_matrix, cmap='YlOrRd', aspect='auto')
        
        # 设置坐标轴（使用负数时间轴）
        tick_positions = np.arange(0, n_days, max(1, n_days // 10))
        tick_labels = [-i-1 for i in tick_positions]  # 转换为负数标签
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_yticks(np.arange(len(lambda_values)))
        ax.set_yticklabels([f'λ={λ}' for λ in lambda_values])
        
        ax.set_xlabel('时间 (天)', fontsize=12)
        ax.set_ylabel('Lambda值', fontsize=12)
        ax.set_title(f'EWMA权重热力图 (N = {n_days}天)', fontsize=14)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('权重', fontsize=12)
        
        plt.tight_layout()
        plt.show()


def main():
    """
    主函数：演示EWMA权重可视化
    """
    # 创建可视化对象
    visualizer = EWMAWeightVisualizer()
    
    # 设置参数
    n_days = 100  # 历史数据天数
    lambda_values = [0.94, 0.90, 0.84, 0.80, 0.70]  # 不同的lambda值
    
    print("EWMA权重可视化分析")
    print("=" * 50)
    
    # 1. 绘制多个lambda值的权重分布对比
    print("1. 绘制权重分布对比图...")
    visualizer.plot_multiple_lambda_weights(n_days, lambda_values)
    
    # 2. 分析权重特征
    print("2. 分析权重特征...")
    analysis_df = visualizer.analyze_weight_characteristics(n_days, lambda_values)
    print(analysis_df.round(4))
    
    # 3. 绘制权重热力图
    print("3. 绘制权重热力图...")
    visualizer.plot_weight_heatmap(n_days, lambda_values)
    
    # 4. 单独展示某个lambda值的详细权重分布
    print("4. 展示λ=0.94的详细权重分布...")
    fig, ax = plt.subplots(figsize=(12, 6))
    visualizer.plot_single_lambda_weights(50, 0.94, ax)
    plt.show()
    
    print("\n分析完成！")


if __name__ == "__main__":
    visualizer = EWMAWeightVisualizer()
    visualizer.plot_multiple_lambda_weights(100, [0.94, 0.90, 0.84, 0.80])