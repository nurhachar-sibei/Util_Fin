'''
本模块用于对多事件序列进行主成分分析（PCA）
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Optional, Union, List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class PCAAnalyzer:
    """
    多事件序列PCA分解分析器
    
    该类提供了对多个时间序列进行主成分分析的完整功能，包括：
    - 数据预处理和标准化
    - PCA分解计算
    - 各项PCA指标计算和展示
    - 可视化分析
    - 结果导出
    """
    
    def __init__(self, standardize: bool = True, n_components: Optional[int] = None):
        """
        初始化PCA分析器
        
        Parameters:
        -----------
        standardize : bool, default=True
            是否对数据进行标准化处理
        n_components : int, optional
            主成分数量，如果为None则保留所有主成分
        """
        self.standardize = standardize
        self.n_components = n_components
        self.scaler = StandardScaler() if standardize else None
        self.pca = None
        self.data = None
        self.data_scaled = None
        self.feature_names = None
        self.is_fitted = False
        
    def fit(self, data: Union[pd.DataFrame, np.ndarray], feature_names: Optional[List[str]] = None) -> 'PCAAnalyzer':
        """
        拟合PCA模型
        
        Parameters:
        -----------
        data : pd.DataFrame or np.ndarray
            输入数据，行为观测值，列为特征（事件序列）
        feature_names : list of str, optional
            特征名称列表，如果data是DataFrame则自动获取
            
        Returns:
        --------
        self : PCAAnalyzer
            返回自身以支持链式调用
        """
        # 数据预处理
        if isinstance(data, pd.DataFrame):
            self.data = data.copy()
            self.feature_names = list(data.columns) if feature_names is None else feature_names
            data_array = data.values
        else:
            self.data = pd.DataFrame(data)
            self.feature_names = feature_names or [f'Feature_{i}' for i in range(data.shape[1])]
            data_array = data
            
        # 检查数据有效性
        if np.any(np.isnan(data_array)) or np.any(np.isinf(data_array)):
            raise ValueError("数据包含NaN或无穷值，请先进行数据清洗")
            
        # 标准化处理
        if self.standardize:
            self.data_scaled = self.scaler.fit_transform(data_array)
        else:
            self.data_scaled = data_array.copy()
            
        # 设置主成分数量
        n_features = data_array.shape[1]
        if self.n_components is None:
            self.n_components = n_features
        else:
            self.n_components = min(self.n_components, n_features)
            
        # 拟合PCA模型
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(self.data_scaled)
        
        self.is_fitted = True
        return self
    
    def transform(self, data: Optional[Union[pd.DataFrame, np.ndarray]] = None) -> np.ndarray:
        """
        将数据转换到主成分空间
        
        Parameters:
        -----------
        data : pd.DataFrame or np.ndarray, optional
            要转换的数据，如果为None则使用训练数据
            
        Returns:
        --------
        np.ndarray
            转换后的主成分数据
        """
        if not self.is_fitted:
            raise ValueError("模型尚未拟合，请先调用fit方法")
            
        if data is None:
            data_to_transform = self.data_scaled
        else:
            if isinstance(data, pd.DataFrame):
                data_array = data.values
            else:
                data_array = data
                
            if self.standardize:
                data_to_transform = self.scaler.transform(data_array)
            else:
                data_to_transform = data_array
                
        return self.pca.transform(data_to_transform)
    
    def get_explained_variance_ratio(self) -> np.ndarray:
        """
        获取解释方差比例
        
        Returns:
        --------
        np.ndarray
            每个主成分的解释方差比例
        """
        if not self.is_fitted:
            raise ValueError("模型尚未拟合，请先调用fit方法")
        return self.pca.explained_variance_ratio_
    
    def get_cumulative_variance_ratio(self) -> np.ndarray:
        """
        获取累积解释方差比例
        
        Returns:
        --------
        np.ndarray
            累积解释方差比例
        """
        if not self.is_fitted:
            raise ValueError("模型尚未拟合，请先调用fit方法")
        return np.cumsum(self.pca.explained_variance_ratio_)
    
    def get_components_matrix(self) -> pd.DataFrame:
        """
        获取主成分系数矩阵（载荷矩阵）
        
        Returns:
        --------
        pd.DataFrame
            主成分系数矩阵，行为主成分，列为原始特征
        """
        if not self.is_fitted:
            raise ValueError("模型尚未拟合，请先调用fit方法")
            
        components_df = pd.DataFrame(
            self.pca.components_,
            columns=self.feature_names,
            index=[f'PC{i+1}' for i in range(self.n_components)]
        )
        return components_df
    
    def get_feature_contributions(self, pc_index: int = 0) -> pd.Series:
        """
        获取指定主成分中各特征的贡献度
        
        Parameters:
        -----------
        pc_index : int, default=0
            主成分索引（从0开始）
            
        Returns:
        --------
        pd.Series
            各特征对指定主成分的贡献度（绝对值）
        """
        if not self.is_fitted:
            raise ValueError("模型尚未拟合，请先调用fit方法")
            
        if pc_index >= self.n_components:
            raise ValueError(f"主成分索引超出范围，最大索引为{self.n_components-1}")
            
        contributions = np.abs(self.pca.components_[pc_index])
        return pd.Series(contributions, index=self.feature_names, name=f'PC{pc_index+1}_Contribution')
    
    def get_eigenvalues(self) -> np.ndarray:
        """
        获取特征值
        
        Returns:
        --------
        np.ndarray
            各主成分对应的特征值
        """
        if not self.is_fitted:
            raise ValueError("模型尚未拟合，请先调用fit方法")
        return self.pca.explained_variance_
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """
        获取PCA分析的汇总统计信息
        
        Returns:
        --------
        pd.DataFrame
            包含特征值、解释方差比例、累积方差比例等信息的汇总表
        """
        if not self.is_fitted:
            raise ValueError("模型尚未拟合，请先调用fit方法")
            
        summary = pd.DataFrame({
            '主成分': [f'PC{i+1}' for i in range(self.n_components)],
            '特征值': self.get_eigenvalues(),
            '解释方差比例': self.get_explained_variance_ratio(),
            '累积方差比例': self.get_cumulative_variance_ratio()
        })
        
        return summary
    
    def find_optimal_components(self, variance_threshold: float = 0.95) -> int:
        """
        根据累积方差比例阈值确定最优主成分数量
        
        Parameters:
        -----------
        variance_threshold : float, default=0.95
            累积方差比例阈值
            
        Returns:
        --------
        int
            建议的主成分数量
        """
        if not self.is_fitted:
            raise ValueError("模型尚未拟合，请先调用fit方法")
            
        cumulative_variance = self.get_cumulative_variance_ratio()
        optimal_n = np.argmax(cumulative_variance >= variance_threshold) + 1
        return optimal_n
    
    def plot_explained_variance(self, figsize: Tuple[int, int] = (12, 5)) -> None:
        """
        绘制解释方差比例图
        
        Parameters:
        -----------
        figsize : tuple, default=(12, 5)
            图形大小
        """
        if not self.is_fitted:
            raise ValueError("模型尚未拟合，请先调用fit方法")
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 解释方差比例柱状图
        pc_labels = [f'PC{i+1}' for i in range(self.n_components)]
        ax1.bar(pc_labels, self.get_explained_variance_ratio())
        ax1.set_title('各主成分解释方差比例')
        ax1.set_xlabel('主成分')
        ax1.set_ylabel('解释方差比例')
        ax1.tick_params(axis='x', rotation=45)
        
        # 累积解释方差比例折线图
        ax2.plot(pc_labels, self.get_cumulative_variance_ratio(), 'bo-')
        ax2.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95%阈值')
        ax2.set_title('累积解释方差比例')
        ax2.set_xlabel('主成分')
        ax2.set_ylabel('累积解释方差比例')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_components_heatmap(self, figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        绘制主成分载荷矩阵热力图
        
        Parameters:
        -----------
        figsize : tuple, default=(10, 8)
            图形大小
        """
        if not self.is_fitted:
            raise ValueError("模型尚未拟合，请先调用fit方法")
            
        components_df = self.get_components_matrix()
        
        plt.figure(figsize=figsize)
        sns.heatmap(components_df, annot=True, cmap='RdBu_r', center=0, 
                   fmt='.3f', cbar_kws={'label': '载荷系数'})
        plt.title('主成分载荷矩阵热力图')
        plt.xlabel('原始特征')
        plt.ylabel('主成分')
        plt.tight_layout()
        plt.show()
    
    def plot_feature_contributions(self, pc_index: int = 0, top_n: int = None, 
                                 figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        绘制指定主成分的特征贡献度图
        
        Parameters:
        -----------
        pc_index : int, default=0
            主成分索引
        top_n : int, optional
            显示前N个贡献度最高的特征，如果为None则显示所有特征
        figsize : tuple, default=(10, 6)
            图形大小
        """
        if not self.is_fitted:
            raise ValueError("模型尚未拟合，请先调用fit方法")
            
        contributions = self.get_feature_contributions(pc_index)
        
        if top_n is not None:
            contributions = contributions.nlargest(top_n)
            
        plt.figure(figsize=figsize)
        contributions.plot(kind='bar')
        plt.title(f'PC{pc_index+1} 特征贡献度')
        plt.xlabel('特征')
        plt.ylabel('贡献度（绝对值）')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_biplot(self, pc1: int = 0, pc2: int = 1, figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        绘制PCA双标图（biplot）
        
        Parameters:
        -----------
        pc1 : int, default=0
            第一个主成分索引
        pc2 : int, default=1
            第二个主成分索引
        figsize : tuple, default=(10, 8)
            图形大小
        """
        if not self.is_fitted:
            raise ValueError("模型尚未拟合，请先调用fit方法")
            
        if max(pc1, pc2) >= self.n_components:
            raise ValueError("主成分索引超出范围")
            
        # 获取转换后的数据
        transformed_data = self.transform()
        
        # 获取载荷向量
        loadings = self.pca.components_[[pc1, pc2]].T
        
        plt.figure(figsize=figsize)
        
        # 绘制数据点
        plt.scatter(transformed_data[:, pc1], transformed_data[:, pc2], alpha=0.6)
        
        # 绘制载荷向量
        for i, (feature, loading) in enumerate(zip(self.feature_names, loadings)):
            plt.arrow(0, 0, loading[0]*3, loading[1]*3, 
                     head_width=0.1, head_length=0.1, fc='red', ec='red')
            plt.text(loading[0]*3.2, loading[1]*3.2, feature, 
                    fontsize=10, ha='center', va='center')
        
        plt.xlabel(f'PC{pc1+1} ({self.get_explained_variance_ratio()[pc1]:.1%})')
        plt.ylabel(f'PC{pc2+1} ({self.get_explained_variance_ratio()[pc2]:.1%})')
        plt.title('PCA双标图')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def export_results(self, filepath: str, include_transformed_data: bool = True) -> None:
        """
        导出PCA分析结果到Excel文件
        
        Parameters:
        -----------
        filepath : str
            输出文件路径
        include_transformed_data : bool, default=True
            是否包含转换后的主成分数据
        """
        if not self.is_fitted:
            raise ValueError("模型尚未拟合，请先调用fit方法")
            
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # 汇总统计
            self.get_summary_statistics().to_excel(writer, sheet_name='汇总统计', index=False)
            
            # 主成分载荷矩阵
            self.get_components_matrix().to_excel(writer, sheet_name='载荷矩阵')
            
            # 原始数据
            if isinstance(self.data, pd.DataFrame):
                self.data.to_excel(writer, sheet_name='原始数据')
            
            # 转换后的主成分数据
            if include_transformed_data:
                transformed_data = self.transform()
                pc_columns = [f'PC{i+1}' for i in range(self.n_components)]
                transformed_df = pd.DataFrame(transformed_data, columns=pc_columns)
                if hasattr(self.data, 'index'):
                    transformed_df.index = self.data.index
                transformed_df.to_excel(writer, sheet_name='主成分数据')
    
    def get_reconstruction_error(self, n_components: Optional[int] = None) -> float:
        """
        计算重构误差
        
        Parameters:
        -----------
        n_components : int, optional
            用于重构的主成分数量，如果为None则使用所有主成分
            
        Returns:
        --------
        float
            重构误差（均方误差）
        """
        if not self.is_fitted:
            raise ValueError("模型尚未拟合，请先调用fit方法")
            
        if n_components is None:
            n_components = self.n_components
            
        # 使用指定数量的主成分进行重构
        pca_temp = PCA(n_components=n_components)
        pca_temp.fit(self.data_scaled)
        
        # 转换和逆转换
        transformed = pca_temp.transform(self.data_scaled)
        reconstructed = pca_temp.inverse_transform(transformed)
        
        # 计算重构误差
        mse = np.mean((self.data_scaled - reconstructed) ** 2)
        return mse
    
    def __repr__(self) -> str:
        """返回对象的字符串表示"""
        if self.is_fitted:
            return (f"PCAAnalyzer(n_components={self.n_components}, "
                   f"standardize={self.standardize}, fitted=True)")
        else:
            return (f"PCAAnalyzer(n_components={self.n_components}, "
                   f"standardize={self.standardize}, fitted=False)")