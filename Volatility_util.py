import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

from arch import arch_model
from pypfopt.risk_models  import semicovariance

from sklearn.covariance import LedoitWolf
from scipy import stats

class Cov_Matrix():
    def __init__(self,ret,method):
        self.cal_df = ret
        self.method = method
    
    #协方差计算方法：
    #样本协方差计算
    def calculate_cov_matrix(self,df,frequency):
        """计算协方差矩阵"""
        one_cov_matrix = df.cov()*frequency #年化协方差
        return np.matrix(one_cov_matrix)
    #半衰协方差
    def calculate_half_cov_matrix(self,df,frequency):
        #计算半衰协方差矩阵
        length_cal_df = len(df)
        length_sub_df = int(length_cal_df/4)
        train_subset = df.iloc[0:length_sub_df]
        cov_matrix = self.calculate_cov_matrix(train_subset,frequency)
        for i in range(1,4):
            train_subset = df.iloc[i*length_sub_df:(i+1)*length_sub_df]
            sub_cov_matrix = self.calculate_cov_matrix(train_subset,frequency)
            if i==1:
                sub_cov_matrix = sub_cov_matrix*2/10
            elif i==2:
                sub_cov_matrix = sub_cov_matrix*3/10
            else:
                sub_cov_matrix = sub_cov_matrix*4/10
            cov_matrix = cov_matrix+sub_cov_matrix
        return np.matrix(cov_matrix)
    #对角矩阵方差
    def calculate_diag_cov_matrix(self,df,frequency):
        one_cov_matrix = np.matrix(df.cov()*frequency)
        diag_matrix = np.diagflat(np.diag(one_cov_matrix))
        return np.matrix(diag_matrix)
    def calculate_spring_cov_matrix(self,df,frequency):
        lw = LedoitWolf()
        lw.fit(df)
        shrink_cov = np.matrix(lw.covariance_)
        return np.matrix(shrink_cov*frequency)
    def calculate_threshold_cov_matrix(self,df,frequency,threshold=0.03):
        sample_cov_matrix = self.calculate_cov_matrix(df,frequency)
        cov_matrix = np.array(sample_cov_matrix)
        abs_cov = np.abs(cov_matrix) 
        cov_thresh = cov_matrix * (abs_cov > threshold ) 
        np.fill_diagonal(cov_thresh,  np.diag(cov_matrix))   # 保留对角线方差 
        return np.matrix(cov_thresh) 
    def calculate_EWMA_cov_matrix(self,df,frequency,lambda_=0.84):
        # 设置衰减因子lambda（常用0.94）
        lambda_ = lambda_
        # print(lambda_)
        # 初始化协方差矩阵（使用第一个时间点的收益率乘积作为初始协方差）
        n_assets = df.shape[1] 
        cov_matrix = np.zeros((n_assets,  n_assets))
        # 第一个时间点：使用收益率乘积（注意：这里假设第一个时间点没有前一个协方差，因此用第一个观测值构造）
        # 或者我们可以用历史样本协方差初始化，但这里简单处理
        # 更稳健的做法是用前一段时间（如30天）的样本协方差初始化
        initial_cov = np.outer(df.iloc[0],  df.iloc[0]) 
        cov_series = [initial_cov]  # 存储每个时间点的协方差矩阵
        # 迭代计算EWMA条件协方差矩阵（从第二个时间点开始）
        for i in range(1, len(df)):
            r_t = df.iloc[i]   # 当前收益率向量
            prev_cov = cov_series[-1]  # 前一期协方差矩阵
            # 更新协方差矩阵
            new_cov = (1 - lambda_) * np.outer(r_t,  r_t) + lambda_ * prev_cov
            cov_series.append(new_cov) 
        # 将结果转换为DataFrame（以最后一个时间点为例）
        EWMA_cov = cov_series[-1]
        return np.matrix(EWMA_cov)*frequency
    def calculate_GARCH_cov_matrix(self,df,frequency):
        ret_ = df
        returns_centered = ret_ - ret_.mean() 
        # 2. 拟合单资产 GARCH(1,1)
        cond_variances = pd.DataFrame()
        for asset in returns_centered.columns: 
            model = arch_model(returns_centered[asset], vol='Garch', p=1, q=1)
            result = model.fit(update_freq=0,  disp='off')
            cond_variances[asset] = result.conditional_volatility**2
        #3. 计算恒定相关系数矩阵 R
        std_resid = returns_centered / np.sqrt(cond_variances) 
        P = std_resid.corr()
        # 4. 计算时变条件协方差 D_t
        H_t = np.zeros((len(ret_),  len(ret_.columns), len(ret_.columns)))
        for t in range(len(ret_)):
            D_t = np.diag(np.sqrt(cond_variances.iloc[t].values)) 
            H_t[t] = D_t @ P.values  @ D_t
        return np.matrix(H_t[-1])*frequency
    def calculate_semi_cov_matrix(self,df,frequency):
        ret_ = df 
        pv = (ret_+1).cumprod()
        semi_cov = semicovariance(pv,benchmark=0,frequency=frequency) #注意simi函数中输入的是price
        return  np.matrix(semi_cov)
    def calculate_EWMA_Semi_cov_matrix(self,ret_,frequency):   
        def _is_positive_semidefinite(matrix):
            try:
                np.linalg.cholesky(matrix + 1e-16 * np.eye(len(matrix)))
                return True
            except np.linalg.LinAlgError:
                return False
        def fix_nonpositive_semidefinite(matrix, fix_method="spectral"):
            if _is_positive_semidefinite(matrix):
                return matrix
            warnings.warn(
                "The covariance matrix is non positive semidefinite. Amending eigenvalues."
            )

            q, V = np.linalg.eigh(matrix)
            if fix_method == "spectral":
                q = np.where(q > 0, q, 0)
                fixed_matrix = V @ np.diag(q) @ V.T
            elif fix_method == "diag":
                min_eig = np.min(q)
                fixed_matrix = matrix - 1.1 * min_eig * np.eye(len(matrix))
            else:
                raise NotImplementedError("Method {} not implemented".format(fix_method))

            if not _is_positive_semidefinite(fixed_matrix):  
                warnings.warn(
                    "Could not fix matrix. Please try a different risk model.", UserWarning
                )

            if isinstance(matrix, pd.DataFrame):
                tickers = matrix.index
                return pd.DataFrame(fixed_matrix, index=tickers, columns=tickers)
            else:
                return fixed_matrix
        
        drops = np.fmin(ret_ - 0, 0)
        M_1 = self.calculate_EWMA_cov_matrix(drops,frequency)
        M_1 = fix_nonpositive_semidefinite(M_1)
        return np.matrix(M_1)
    #协方差计算方式选择
    def calculate_cal_cov_matrix(self,frequency=252):
        method = self.method
        if method == 'ALL': #ALL 表示普通协方差
            self.cal_cov_matrix = self.calculate_cov_matrix(self.cal_df,frequency=frequency)
        elif method=='HALF': #HALF 表示半衰协方差
            self.cal_cov_matrix = self.calculate_half_cov_matrix(self.cal_df,frequency=frequency)
        elif method=='DIAG': #DIAG表示对角协方差
            self.cal_cov_matrix = self.calculate_diag_cov_matrix(self.cal_df,frequency=frequency)
        elif method=='SPRING':
            self.cal_cov_matrix = self.calculate_spring_cov_matrix(self.cal_df,frequency=frequency)
        elif method=='THRESH':
            self.cal_cov_matrix = self.calculate_threshold_cov_matrix(self.cal_df,frequency=frequency)
        elif method=='EWMA':
            self.cal_cov_matrix = self.calculate_EWMA_cov_matrix(self.cal_df,frequency=frequency)
        elif method=='GARCH':
            self.cal_cov_matrix = self.calculate_GARCH_cov_matrix(self.cal_df,frequency=frequency)
        elif method=='SEMI':
            self.cal_cov_matrix = self.calculate_semi_cov_matrix(self.cal_df,frequency=frequency)
        elif method=='EWMA_SEMI':
            self.cal_cov_matrix = self.calculate_EWMA_Semi_cov_matrix(self.cal_df,frequency=frequency)
        return self.cal_cov_matrix