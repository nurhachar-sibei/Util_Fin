'''
本模块用于计算由于价格变动而带来的投资组合内部各资产权重变动
'''
import pandas as pd
import numpy as np


def calculate_dynamic_weight(hold_df, init_weight, init_date):
    """
    计算单期因价格变动产生的动态权重
    
    Parameters:
    -----------
    hold_df : pd.DataFrame
        持有期内所有资产的收益率表格
    init_weight : np.matrix
        第一天所持有的权重(n×1矩阵)
    init_date : pd.Timestamp
        第一天所对应的日期
        
    Returns:
    --------
    weights_df : pd.DataFrame
        动态权重时间序列
    """
    weights_df = pd.DataFrame(
        index=hold_df.index, 
        columns=hold_df.columns
    ).fillna(0)
    
    # 设置初始权重
    weights_df.loc[init_date] = init_weight.T.tolist()[0]
    prev_weights = weights_df.loc[init_date].values
    
    # 逐日更新权重
    for i in range(1, len(weights_df)):
        daily_ret = hold_df.iloc[i].values
        # 计算组合总收益率
        port_return = np.dot(prev_weights, daily_ret)
        # 计算新权重
        new_weights = prev_weights * (1 + daily_ret) / (1 + port_return)
        weights_df.iloc[i] = new_weights
        prev_weights = new_weights
    
    return weights_df


def calculate_dynamic_weight_series(self, hold_total_df, weight_df):
    """
    计算多期因价格变动产生的动态权重
    
    Parameters:
    -----------
    hold_total_df : pd.DataFrame
        持有期内所有资产的收益率表格
    weight_df : np.matrix
        换仓日所持有的权重(n×1矩阵)

    Returns:
    --------
    weights_dy_df : pd.DataFrame
        动态权重时间序列
    """
    daily_weight = None
    for i in range(len(weights_df)):
        date = weights_df.index[i]
        try:
            next_date = weights_df.index[i+1]
        except:
            next_date = weight_df.index[-1]
        hold_df = hold_total_df.loc[date:next_date].iloc[:-1]

        # 计算动态权重
        daily_weight_sub = calculate_dynamic_weight(
            hold_df, weight_df.loc[date].values, date
        )
        if daily_weight is None:
            daily_weight = daily_weight_sub
        else:
            daily_weight = pd.concat([daily_weight, daily_weight_sub])
        
    return daily_weight


