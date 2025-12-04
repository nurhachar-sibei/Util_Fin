#!/usr/bin/env python
# coding: utf-8
'''
本模块用于从Wind数据库中获取股票、基金、指数等资产的历史价格数据,目前已经暂停更新，请基于easy_manager 编写更加自由的更新脚本
'''

from operator import index
from turtle import update
from WindPy import w
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import numpy as np
from datetime import datetime
w.start()



#code_price_get
def get_hfq_price(code_list,start_date,end_date):
    code_text = ''
    for i in code_list:
        code_text = code_text+i+','
    content_request = w.wsd(code_text,"close",start_date,end_date,"PriceAdj=B")
    price_df  = pd.DataFrame(index=content_request.Times,
                             columns=content_request.Codes,
                             data=np.array(content_request.Data).T)
    price_df.index= pd.to_datetime(price_df.index)
    # price_df.to_excel(file_name,index_label='date')
    return price_df
def get_daily_info(code_list,factor,start_date,end_date):
    code_text = ''
    for i in code_list:
        code_text = code_text+i+','
    content_request = w.wsd(code_text, factor, start_date, end_date, "unit=1;PriceAdj=B")
    price_df  = pd.DataFrame(index=content_request.Times,
                             columns=content_request.Codes,
                             data=np.array(content_request.Data).T)
    price_df.index= pd.to_datetime(price_df.index)
    # price_df.to_excel(file_name,index_label='date')
    return price_df

def update_hfq_price(file_name):
    price_df = pd.read_excel(file_name,index_col=0)
    price_df.index = pd.to_datetime(price_df.index,format='%Y-%m-%d')
    file_last_date = price_df.index[-1].strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%Y-%m-%d')   
    code_list = price_df.columns
    start_date = file_last_date
    end_date = current_time
    if current_time != file_last_date:
        new_price_df = get_hfq_price(code_list,start_date,end_date)
        price_df = pd.concat([price_df,new_price_df],axis=0)
    price_df.to_excel(file_name)

def add_new_asset(code_list,file_name):
    price_df = pd.read_excel(file_name,index_col=0)
    price_df.index = pd.to_datetime(price_df.index,format='%Y-%m-%d')
    add_code_list = list(set(code_list) - set(price_df.columns))
    new_price_df = get_hfq_price(add_code_list,price_df.index[0],price_df.index[-1])
    new_price_df.index = pd.to_datetime(new_price_df.index)
    price_df = pd.concat([price_df,new_price_df],axis=1)
    price_df.to_excel(file_name)

def get_workspace_data(file_name,code_list,start_date,end_date = datetime.now().strftime('%Y-%m-%d') ):
    price_df = pd.read_excel(file_name,index_col=0)
    price_df.index = pd.to_datetime(price_df.index,format='%Y-%m-%d')
    price_df = price_df[code_list]
    price_df = price_df.loc[start_date:end_date]
    return price_df



# 价格标准缩放(以一百为起点)
def price_scale(price_df):
    price_scale_df = price_df/price_df.iloc[0] * 100
    return price_scale_df



# 收益率获取
def get_return_df(price_df):
    ret_ = price_df.pct_change()
    ret_ = ret_.fillna(0)
    return ret_



#折现图作图
def fin_plot_line(data):
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1,1,figsize=(18, 6))
    sns.set(palette="muted")
    plt.plot(data, label=data.columns, alpha=.6)
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  
    plt.legend()
    plt.show()

#柱状图作图
def fin_plot_hist(data):
    hist = data.hist(bins=20, figsize=(12, 4))
    plt.show()
    
#热力图：
def fin_plot_corr(data):
    corr = data.corr()
    corr_heatmap = sns.heatmap(corr, cmap="YlGnBu", linewidths=.2)
    plt.show()


if __name__ == '__main__':
    # code_list = ['000001.SH','000002.SH','000003.SH','000004.SH','000005.SH']
    # start_date = '2020-01-01'
    # end_date = '2022-12-31'
    # price_df = get_hfq_price(code_list,start_date,end_date)
    # price_df = pd.to_excel("price.xlsx",price_df) 
    # print(price_df)
    update_hfq_price('price.xlsx')
    add_new_asset(['000006.SH'],'price.xlsx')
    price_df = get_workspace_data('price.xlsx',['000001.SH','000002.SH','000003.SH','000004.SH','000005.SH','000006.SH'],'2020-01-01')
    print(price_df)


