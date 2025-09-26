import pandas as pd
import numpy as np
from datetime import datetime
import os
import Wind_util
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

#整体评价
def max_draw_down(daily_return): 
    cumulative_wealth = (1 + daily_return).cumprod()
    running_max = cumulative_wealth.expanding().max()
    drawdown = (cumulative_wealth - running_max) / running_max
    max_drawdown = drawdown.min()
    return max_drawdown
#各类指标
def get_eval_indicator(pv):
    """各种评价指标"""
    eval_indicator = pd.Series(index=['累计收益率','年化收益率','年化波动率','最大回撤','sharpe比率','Calmar比率'])
    return_df = Wind_util.get_return_df(pv)
    #累计收益率
    eval_indicator['累计收益率'] = pv.iloc[-1]/pv.iloc[0] - 1
    #年化收益率
    annual_ret = (pv.iloc[-1]/pv.iloc[0])**(252/pv.shape[0])-1
    eval_indicator['年化收益率'] = annual_ret
    #annual_ret = np.power(1+return_df.mean(), 250)-1 # 几何年化收益
    #年化波动率
    sigma = return_df.std() * (252**0.5)
    eval_indicator['年化波动率'] = sigma
    # 最大回撤
    dd = max_draw_down(return_df)
    eval_indicator['最大回撤'] = dd
    #夏普比率  无风险利率是0%
    bench_annual_ret = 0
    sharpe = (annual_ret-bench_annual_ret)/sigma if sigma!=0 else 0
    eval_indicator['sharpe比率'] = sharpe
    #Calmar比率=年化收益率/最大历史回撤
    calmar = annual_ret/abs(dd) if abs(dd)!=0 else 0
    eval_indicator['Calmar比率'] = calmar
    return eval_indicator
#生成结果
def get_eval_portfolio(pv_df,title):
    """返回各个模型回测的评价结果"""
    pv_timeseries = pd.DataFrame(pv_df)
    pv_timeseries.columns = [title]
    eval_portfolio = pd.DataFrame(columns=pv_timeseries.columns)

    for name in eval_portfolio.columns:
        pv_sub = pv_timeseries[name]
        eval_portfolio[name] = get_eval_indicator(pv_sub)
    return eval_portfolio


#分年度评价
def calculate_rolling_cumulative_returns(returns, window=250):
    """
    计算滚动累计收益率
    每一日计算过去250天的累计收益率
    """
    rolling_cum_returns = []
    
    for i in range(len(returns)):
        if i < window - 1:
            # 样本不足250天，设为NaN
            rolling_cum_returns.append(np.nan)
        else:
            # 计算过去250天的累计收益率
            period_returns = returns.iloc[i-window+1:i+1]
            cum_return = (1 + period_returns).prod() - 1
            rolling_cum_returns.append(cum_return)
    
    return pd.Series(rolling_cum_returns, index=returns.index)

def calculate_past_250_drawdown(returns_series):
    """
    计算每日过去250天的回撤率past_250_draw
    """
    past_250_draw = []
    dates = []
    
    for i in range(len(returns_series)):
        # 获取当前日期
        current_date = returns_series.index[i]
        dates.append(current_date)
        
        # 获取过去250天的样本（包括当天）
        start_idx = max(0, i - 249)
        if i - start_idx + 1 < 250:
            # 样本量小于250，设为NaN
            past_250_draw.append(np.nan)
        else:
            # 获取250天的收益率样本
            sample_returns = returns_series.iloc[start_idx:i+1]
            
            # 计算累计净值
            cumulative_values = (1 + sample_returns).cumprod()
            
            # 计算第一天到最低点的回撤率
            first_day_value = cumulative_values.iloc[0]
            min_value = cumulative_values.min()
            drawdown = (min_value - first_day_value) / first_day_value
            
            past_250_draw.append(drawdown)
    
    # 创建DataFrame
    draw_df = pd.DataFrame({
        'date': dates,
        'past_250_draw': past_250_draw
    })
    
    return draw_df

def calculate_var_historical(past_250_draw_sample, confidence_level=0.95):
    """
    使用历史数据法计算VaR
    """
    if len(past_250_draw_sample) < 500:
        return np.nan
    
    # 去除NaN值
    valid_sample = past_250_draw_sample.dropna()
    
    if len(valid_sample) < 500:
        return np.nan
    
    # 计算VaR（损失的分位数）
    var_value = np.percentile(valid_sample, (1 - confidence_level) * 100)
    
    return var_value

def calculate_annual_metrics_with_var(returns_df,dafult_VaR_year_windows):
    """
    计算年度风险收益指标，包括预计VaR
    """
    # 确保日期列为datetime格式
    if 'date' in returns_df.columns:
        returns_df['date'] = pd.to_datetime(returns_df['date'])
        returns_df.set_index('date', inplace=True)
    elif returns_df.index.dtype != 'datetime64[ns]':
        returns_df.index = pd.to_datetime(returns_df.index)
    
    # 获取收益率列（假设第一列是收益率）
    returns_col = returns_df.columns[0]
    returns = returns_df[returns_col]
    
    # 计算每日过去250天的回撤率
    print("正在计算每日过去250天的回撤率...")
    draw_df = calculate_past_250_drawdown(returns)
    
    # 保存回撤数据到draw.xlsx
    draw_df.to_excel('./excel/draw.xlsx', index=False)
    print("回撤数据已保存到draw.xlsx")
    
    # 按年分组
    annual_results = []
    years = sorted(returns.index.year.unique())
    
    for year in years:
        print(f"正在分析 {year} 年...")
        year_returns = returns[returns.index.year == year]
        
        if len(year_returns) == 0:
            continue
            
        # 1.1 年化收益率（区间收益率）
        cumulative_return = (1 + year_returns).prod() - 1
        trading_days = len(year_returns)
        annualized_return = (1 + cumulative_return)- 1




        # 2. 最大回撤
        cumulative_wealth = (1 + year_returns).cumprod()
        running_max = cumulative_wealth.expanding().max()
        drawdown = (cumulative_wealth - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 2.1 计算年初到最低点回撤
        year_start_value = 1.0  # 年初净值设为1
        cumulative_returns = (1 + year_returns).cumprod()
        min_value = cumulative_returns.min()  # 年内最低净值
        year_start_to_min_drawdown = (min_value - year_start_value) / year_start_value
        # 如果回撤大于0，则设为0
        year_start_to_min_drawdown = min(year_start_to_min_drawdown, 0)
        
        # 基于年初到最低点回撤计算最大投资额度
        max_loss_limit_drawdown = 200000000  # 2亿最大亏损额
        if year_start_to_min_drawdown == 0:
            max_investment_by_drawdown = "无限制"  # 回撤为0时无投资额度限制
        else:
            max_investment_by_drawdown = max_loss_limit_drawdown / abs(year_start_to_min_drawdown)
        
        # 3. 年化波动率
        annualized_volatility = year_returns.std() * np.sqrt(252)
        # 3.1 期间波动率
        period_volatility = year_returns.std() * np.sqrt(len(year_returns))
        
        # 4. 夏普比率（假设无风险利率为0%）
        risk_free_rate = 0
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility != 0 else 0
        
        # 5. 卡玛比率（Calmar Ratio）
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # 6. 预计VaR计算（95%）
        # 使用该年之前的1000个past_250_draw样本
        prior_draw_data = draw_df[draw_df['date'].dt.year < year]['past_250_draw']
        if len(prior_draw_data) >= 252*dafult_VaR_year_windows:
            # 取最近1000个样本
            forecast_sample = prior_draw_data.tail(252*dafult_VaR_year_windows)
        else:
            # 如果不足1000个，使用所有可用样本
            forecast_sample = prior_draw_data
        
        forecast_var_95 = calculate_var_historical(forecast_sample, confidence_level=0.95)
        
        # 7. 计算最大投资规模
        max_loss_limit = 200_000_000  # 2亿
        
        # 预计最大投资规模（95%）
        if pd.notna(forecast_var_95):
            if forecast_var_95 >= 0:
                forecast_max_investment_95 = "无限制"  # VaR为正时无规模限制
            else:
                forecast_max_investment_95 = max_loss_limit / abs(forecast_var_95)
        else:
            forecast_max_investment_95 = np.nan
        forecast_max_investment_95 = float(forecast_max_investment_95)
        max_loss_inyear = forecast_max_investment_95*year_start_to_min_drawdown
        annual_results.append({
            '年度': year,
            '年化收益率': f"{annualized_return:.4f}",
            '年化波动率': f"{annualized_volatility:.4f}",
            '最大回撤': f"{max_drawdown:.4f}",
            '夏普比率': f"{sharpe_ratio:.4f}",
            '卡玛比率': f"{calmar_ratio:.4f}",
            '预计95%VaR': f"{forecast_var_95:.6f}" if pd.notna(forecast_var_95) else "NaN",
            '预计95%投资规模(万元)': f"{forecast_max_investment_95/10000:.2f}" if isinstance(forecast_max_investment_95, (int, float)) and pd.notna(forecast_max_investment_95) else forecast_max_investment_95 if forecast_max_investment_95 == "无限制" else "NaN",
            '年初到最低点回撤': f"{year_start_to_min_drawdown:.4f}",
            "年内最大亏损额(万元)":f"{max_loss_inyear/10000:.2f}" 
        })
    
    return pd.DataFrame(annual_results)

def Year_analysis(ret_df,dafult_VaR_year_windows=5,save_=True):
        # 尝试不同的读取方式
        df = pd.DataFrame(ret_df)        
        print(f"数据读取成功，共{len(df)}行数据")
        print(f"列名: {list(df.columns)}")
        print("\n前5行数据:")
        print(df.head())
        
        # 数据预处理
        # 如果第一列是日期，第二列是收益率
        if len(df.columns) >= 2:
            df.columns = ['date', 'returns'] + list(df.columns[2:])
        elif len(df.columns) == 1:
            # 如果只有一列，假设索引是日期
            df.columns = ['returns']
        
        # 删除空值
        df = df.dropna()
        
        print(f"\n数据清洗后，共{len(df)}行有效数据")
        
        # 计算年度指标（包括VaR分析）
        print("\n开始计算年度风险收益指标和VaR分析...")
        annual_metrics = calculate_annual_metrics_with_var(df,dafult_VaR_year_windows)
        
        # 显示结果
        print("\n=== 策略年度风险收益分析结果（含VaR分析）===")
        # 设置pandas显示选项以获得更好的格式
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 300)
        pd.set_option('display.max_colwidth', 15)
        
        # 确保没有重复行
        annual_metrics = annual_metrics.drop_duplicates().reset_index(drop=True)
        
        # 创建格式化的显示表格
        print("\n年度分析结果:")
        print("-" * 150)
        print(f"{'年度':<6} {'年化收益率':<10} {'最大回撤':<10} {'年初最低回撤':<12} {'基于回撤额度':<15} {'年化波动率':<10} {'夏普比率':<10} {'卡玛比率':<10}")
        print(f"{'':6} {'预计95%VaR':<12} {'预计95%规模':<15}")
        print("-" * 150)
        
        for _, row in annual_metrics.iterrows():
            print(f"{row['年度']:<6} {row['年化收益率']:<10} {row['最大回撤']:<10} {row['年初到最低点回撤']:<12}  {row['年化波动率']:<10} {row['夏普比率']:<10} {row['卡玛比率']:<10}")
            print(f"{'':6} {row['预计95%VaR']:<12} {row['预计95%投资规模(万元)']:<15}")
            print()
        
        print("-" * 150)
        
  

        # 保存结果到Excel
        if save_ == True:
            filename = input("输出文件名:")
            output_file = f'Analysis_Results_{filename.split(".")[0]}.xlsx'
            annual_metrics.to_excel('./excel/'+output_file, index=False)
            print(f"\n结果已保存到: {output_file}")
        
        # 计算整体统计
        print("\n=== 整体统计摘要 ===")
        if len(annual_metrics) > 0:
            # 转换数值列进行统计
            numeric_cols = ['年化收益率', '最大回撤', '年初到最低点回撤', '年化波动率', '夏普比率', '卡玛比率']
            for col in numeric_cols:
                annual_metrics[col] = pd.to_numeric(annual_metrics[col], errors='coerce')
            
            # 计算平均值 - 直接从数值计算，因为数据已经是数值格式
            numeric_cols = ['年化收益率', '最大回撤', '年初到最低点回撤', '年化波动率', '夏普比率', '卡玛比率']
            for col in numeric_cols:
                annual_metrics[col] = pd.to_numeric(annual_metrics[col], errors='coerce')
            
            avg_return = annual_metrics['年化收益率'].mean() * 100
            avg_max_drawdown = annual_metrics['最大回撤'].mean() * 100
            avg_year_start_drawdown = annual_metrics['年初到最低点回撤'].mean() * 100
            avg_volatility = annual_metrics['年化波动率'].mean() * 100
            avg_sharpe = annual_metrics['夏普比率'].mean()
            avg_calmar = annual_metrics['卡玛比率'].mean()
            
            print("\n整体统计摘要:")
            print(f"平均年化收益率: {avg_return:.2f}%")
            print(f"平均最大回撤: {avg_max_drawdown:.2f}%")
            print(f"平均年初到最低点回撤: {avg_year_start_drawdown:.2f}%")
            print(f"平均年化波动率: {avg_volatility:.2f}%")
            print(f"平均夏普比率: {avg_sharpe:.4f}")
            print(f"平均卡玛比率: {avg_calmar:.4f}")
        return(annual_metrics)