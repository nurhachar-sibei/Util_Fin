'''
本模块用于获取调仓日信息、调仓后持有的日期等
'''
import pandas as pd
import numpy as np

class Position_info():
    def __init__(self,total_df,start_date,end_date,change_time_delta,initial_month=1,initial_day=1):
        self.total_df = total_df
        self.start_date = start_date
        self.end_date = end_date
        self.change_time_delta = change_time_delta
        self.stock_names = self.total_df.columns.tolist()
        self.initial_month = initial_month 
        self.initial_day = initial_day

    def position_information(self):
        """
        获取调仓相关信息
        ret_df：全部的收益率信息
        start_date:开仓时间
        end_date:平仓时间
            #注意开仓时间一般不是ret_df的第一个日期，
            #如果指标计算需要用过去240个交易的收益率数据，则开仓时间往往是ret_df240个交易日后的信息
            #前240个需要用于指标计算，再依次滚动
        change_time_delta:调仓间隔
            #该参数提供两种调仓模式：
              #固定日期调仓：在每年，每月，每周，每日第一日进行调仓
              #固定间隔调仓：以日为周期，固定250，120，60，20，5，1日进行调仓
        initial_month:第一个开仓日期的月份
        initial_day:第一个开仓日期的日
        """
        ret_df = self.total_df
        start_date = self.start_date
        end_date = self.end_date
        change_time_delta = self.change_time_delta
        stock_names = self.total_df.columns.tolist()
        initial_month = self.initial_month
        initial_day = self.initial_day
        
        #收益率部分
        # try:
        position_df = ret_df[(ret_df.index>=start_date)&(ret_df.index<=end_date)]
        if str(change_time_delta).isdigit():
            change_position_df = position_df[::change_time_delta]
            change_position_date = change_position_df.index.tolist()

        elif str(change_time_delta) == 'Y':
            # 将输入的月份和日期转换为 datetime 对象  
            input_date = pd.Timestamp(year=2000, month=initial_month, day=initial_day)
            # 提取年份范围  
            years = range(position_df.index.min().year, position_df.index.max().year + 1)

            # 初始化结果  
            change_position_date = []          
            for year in years:  
                # 构造每年的目标日期  
                target_date = pd.Timestamp(year=year, month=initial_month, day=initial_day)
                # 查找该日期或之后最近的日期  
                if target_date in position_df.index:  
                    change_position_date.append(target_date)  
                else:  
                    # 找到之后最近的日期  
                    closest_date = position_df.index[position_df.index > target_date]  
                    if not closest_date.empty:  
                        change_position_date.append(closest_date[0])  
            change_position_df = position_df[position_df.index.isin(change_position_date)]
        elif str(change_time_delta).isalpha():
            change_position_date = position_df.index.to_series().groupby(pd.Grouper(freq=self.change_time_delta)).min() 
            change_position_date = change_position_date.dropna().tolist()
            change_position_df = position_df[position_df.index.isin(change_position_date)]
        return position_df,change_position_df,change_position_date
