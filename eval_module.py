#!/usr/bin/env python
# coding: utf-8
"""
评价模块 - 完全独立的策略评价模块
提供各类评价指标的计算和分析功能
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class PerformanceEvaluator:
    """绩效评价器"""
    
    @staticmethod
    def max_drawdown(returns):
        """
        计算最大回撤
        
        Parameters:
        -----------
        returns : pd.Series
            收益率序列
            
        Returns:
        --------
        max_dd : float
            最大回撤
        """
        cumulative_wealth = (1 + returns).cumprod()
        running_max = cumulative_wealth.expanding().max()
        drawdown = (cumulative_wealth - running_max) / running_max
        max_drawdown = drawdown.min()
        return max_drawdown
    
    @staticmethod
    def calculate_indicators(pv_series, name='策略'):
        """
        计算各类评价指标
        
        Parameters:
        -----------
        pv_series : pd.Series
            净值序列
        name : str
            策略名称
            
        Returns:
        --------
        indicators : pd.Series
            评价指标序列
        """
        indicators = pd.Series(name=name)
        
        # 计算收益率
        returns = pv_series.pct_change().fillna(0)
        
        # 累计收益率
        indicators['累计收益率'] = pv_series.iloc[-1] / pv_series.iloc[0] - 1
        
        # 年化收益率
        trading_days = len(pv_series)
        annual_return = (pv_series.iloc[-1] / pv_series.iloc[0]) ** (252 / trading_days) - 1
        indicators['年化收益率'] = annual_return
        
        # 年化波动率
        annual_volatility = returns.std() * np.sqrt(252)
        indicators['年化波动率'] = annual_volatility
        
        # 最大回撤
        max_dd = PerformanceEvaluator.max_drawdown(returns)
        indicators['最大回撤'] = max_dd
        
        # 夏普比率 (假设无风险利率为0)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
        indicators['夏普比率'] = sharpe_ratio
        
        # Calmar比率
        calmar_ratio = annual_return / abs(max_dd) if abs(max_dd) != 0 else 0
        indicators['Calmar比率'] = calmar_ratio
        
        # Sortino比率 (使用下行波动率)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino_ratio = annual_return / downside_std if downside_std != 0 else 0
        indicators['Sortino比率'] = sortino_ratio
        
        # 最终净值
        indicators['最终净值'] = pv_series.iloc[-1]
        
        return indicators
    
    @staticmethod
    def evaluate_portfolio(pv_series, name='策略'):
        """
        评价组合表现
        
        Parameters:
        -----------
        pv_series : pd.Series
            净值序列
        name : str
            策略名称
            
        Returns:
        --------
        eval_df : pd.DataFrame
            评价结果DataFrame
        """
        indicators = PerformanceEvaluator.calculate_indicators(pv_series, name)
        eval_df = pd.DataFrame(indicators, columns=[name])
        return eval_df
    
    @staticmethod
    def evaluate_multi_portfolios(pv_dict):
        """
        评价多个组合
        
        Parameters:
        -----------
        pv_dict : dict
            净值序列字典 {策略名称: 净值序列}
            
        Returns:
        --------
        eval_df : pd.DataFrame
            多策略评价结果DataFrame
        """
        eval_dfs = []
        
        for name, pv_series in pv_dict.items():
            eval_df = PerformanceEvaluator.evaluate_portfolio(pv_series, name)
            eval_dfs.append(eval_df)
        
        result = pd.concat(eval_dfs, axis=1)
        return result


class RiskAnalyzer:
    """风险分析器"""
    
    @staticmethod
    def calculate_var(returns, confidence_level=0.95, method='historical'):
        """
        计算VaR (Value at Risk)
        
        Parameters:
        -----------
        returns : pd.Series
            收益率序列
        confidence_level : float
            置信水平
        method : str
            计算方法 ('historical', 'parametric')
            
        Returns:
        --------
        var : float
            VaR值
        """
        if method == 'historical':
            # 历史模拟法
            var = np.percentile(returns, (1 - confidence_level) * 100)
        elif method == 'parametric':
            # 参数法(假设正态分布)
            mu = returns.mean()
            sigma = returns.std()
            var = stats.norm.ppf(1 - confidence_level, mu, sigma)
        else:
            raise ValueError("method必须是'historical'或'parametric'")
        
        return var
    
    @staticmethod
    def calculate_cvar(returns, confidence_level=0.95):
        """
        计算CVaR (Conditional Value at Risk) / ES (Expected Shortfall)
        
        Parameters:
        -----------
        returns : pd.Series
            收益率序列
        confidence_level : float
            置信水平
            
        Returns:
        --------
        cvar : float
            CVaR值
        """
        var = RiskAnalyzer.calculate_var(returns, confidence_level, 'historical')
        # CVaR是超过VaR的平均损失
        cvar = returns[returns <= var].mean()
        return cvar
    
    @staticmethod
    def calculate_rolling_drawdown(returns, window=250):
        """
        计算滚动回撤
        
        Parameters:
        -----------
        returns : pd.Series
            收益率序列
        window : int
            滚动窗口长度
            
        Returns:
        --------
        rolling_dd_df : pd.DataFrame
            滚动回撤数据
        """
        rolling_dd = []
        dates = []
        
        for i in range(len(returns)):
            current_date = returns.index[i]
            dates.append(current_date)
            
            start_idx = max(0, i - window + 1)
            if i - start_idx + 1 < window:
                rolling_dd.append(np.nan)
            else:
                sample_returns = returns.iloc[start_idx:i+1]
                cumulative_values = (1 + sample_returns).cumprod()
                first_day_value = cumulative_values.iloc[0]
                min_value = cumulative_values.min()
                drawdown = (min_value - first_day_value) / first_day_value
                rolling_dd.append(drawdown)
        
        rolling_dd_df = pd.DataFrame({
            'date': dates,
            'rolling_drawdown': rolling_dd
        })
        
        return rolling_dd_df


class PeriodAnalyzer:
    """周期分析器 (年度/月度分析)"""
    
    @staticmethod
    def annual_analysis(returns, var_windows=5, max_loss_limit=200_000_000):
        """
        年度分析
        
        Parameters:
        -----------
        returns : pd.Series
            收益率序列
        var_windows : int
            VaR计算的历史窗口年数
        max_loss_limit : float
            最大损失限额
            
        Returns:
        --------
        annual_df : pd.DataFrame
            年度分析结果
        """
        # 确保索引是datetime类型
        if returns.index.dtype != 'datetime64[ns]':
            returns.index = pd.to_datetime(returns.index)
        
        # 计算滚动回撤
        rolling_dd_df = RiskAnalyzer.calculate_rolling_drawdown(returns, window=250)
        
        # 按年分组
        annual_results = []
        years = sorted(returns.index.year.unique())
        
        for year in years:
            year_returns = returns[returns.index.year == year]
            
            if len(year_returns) == 0:
                continue
            
            # 区间收益率
            cumulative_return = (1 + year_returns).prod() - 1
            
            # 最大回撤
            max_dd = PerformanceEvaluator.max_drawdown(year_returns)
            
            # 年初到最低点回撤
            cumulative_values = (1 + year_returns).cumprod()
            min_value = cumulative_values.min()
            year_start_dd = (min_value - 1.0) / 1.0
            year_start_dd = min(year_start_dd, 0)
            
            # 年化波动率
            annual_vol = year_returns.std() * np.sqrt(252)
            
            # 夏普比率
            sharpe = cumulative_return / annual_vol if annual_vol != 0 else 0
            
            # Calmar比率
            calmar = cumulative_return / abs(max_dd) if max_dd != 0 else 0
            
            # 预计VaR (基于历史数据)
            prior_dd = rolling_dd_df[
                rolling_dd_df['date'].dt.year < year
            ]['rolling_drawdown']
            
            if len(prior_dd) >= 252 * var_windows:
                forecast_sample = prior_dd.tail(252 * var_windows)
            else:
                forecast_sample = prior_dd
            
            forecast_var_95 = RiskAnalyzer.calculate_var(
                forecast_sample.dropna(), 
                confidence_level=0.95, 
                method='historical'
            ) if len(forecast_sample.dropna()) >= 500 else np.nan
            
            # 计算投资规模
            if pd.notna(forecast_var_95) and forecast_var_95 < 0:
                forecast_investment = max_loss_limit / abs(forecast_var_95)
                max_loss_in_year = forecast_investment * year_start_dd
            else:
                forecast_investment = np.inf
                max_loss_in_year = np.nan
            
            annual_results.append({
                '年度': year,
                '区间收益率': cumulative_return,
                '年化波动率': annual_vol,
                '最大回撤': max_dd,
                '年初到最低点回撤': year_start_dd,
                '夏普比率': sharpe,
                'Calmar比率': calmar,
                '预计95%VaR': forecast_var_95,
                '预计95%投资规模(万元)': forecast_investment / 10000,
                '年内最大亏损额(万元)': max_loss_in_year / 10000 if pd.notna(max_loss_in_year) else np.nan
            })
        
        annual_df = pd.DataFrame(annual_results)
        return annual_df
    
    @staticmethod
    def monthly_analysis(returns, var_windows=5, max_loss_limit=200_000_000):
        """
        月度分析
        
        Parameters:
        -----------
        returns : pd.Series
            收益率序列
        var_windows : int
            VaR计算的历史窗口年数
        max_loss_limit : float
            最大损失限额
            
        Returns:
        --------
        monthly_df : pd.DataFrame
            月度分析结果
        """
        # 确保索引是datetime类型
        if returns.index.dtype != 'datetime64[ns]':
            returns.index = pd.to_datetime(returns.index)
        
        # 计算滚动回撤
        rolling_dd_df = RiskAnalyzer.calculate_rolling_drawdown(returns, window=250)
        
        # 按月分组
        monthly_results = []
        monthly_groups = returns.groupby([returns.index.year, returns.index.month])
        
        for (year, month), month_returns in monthly_groups:
            if len(month_returns) == 0:
                continue
            
            # 月度收益率
            cumulative_return = (1 + month_returns).prod() - 1
            trading_days = len(month_returns)
            
            # 年化收益率
            if trading_days > 0:
                annual_return = (1 + cumulative_return) ** (252 / trading_days) - 1
            else:
                annual_return = 0
            
            # 最大回撤
            max_dd = PerformanceEvaluator.max_drawdown(month_returns)
            
            # 月初到最低点回撤
            cumulative_values = (1 + month_returns).cumprod()
            min_value = cumulative_values.min()
            month_start_dd = (min_value - 1.0) / 1.0
            month_start_dd = min(month_start_dd, 0)
            
            # 年化波动率
            annual_vol = month_returns.std() * np.sqrt(252)
            
            # 夏普比率
            sharpe = annual_return / annual_vol if annual_vol != 0 else 0
            
            # Calmar比率
            calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
            
            # 预计VaR
            current_date = pd.Timestamp(year, month, 1)
            prior_dd = rolling_dd_df[
                rolling_dd_df['date'] < current_date
            ]['rolling_drawdown']
            
            if len(prior_dd) >= 252 * var_windows:
                forecast_sample = prior_dd.tail(252 * var_windows)
            else:
                forecast_sample = prior_dd
            
            forecast_var_95 = RiskAnalyzer.calculate_var(
                forecast_sample.dropna(), 
                confidence_level=0.95, 
                method='historical'
            ) if len(forecast_sample.dropna()) >= 500 else np.nan
            
            # 计算投资规模
            if pd.notna(forecast_var_95) and forecast_var_95 < 0:
                forecast_investment = max_loss_limit / abs(forecast_var_95)
                max_loss_in_month = forecast_investment * month_start_dd
            else:
                forecast_investment = np.inf
                max_loss_in_month = np.nan
            
            monthly_results.append({
                '年月': f"{year}-{month:02d}",
                '区间收益率': cumulative_return,
                '年化收益率': annual_return,
                '年化波动率': annual_vol,
                '最大回撤': max_dd,
                '月初到最低点回撤': month_start_dd,
                '夏普比率': sharpe,
                'Calmar比率': calmar,
                '预计95%VaR': forecast_var_95,
                '预计95%投资规模(万元)': forecast_investment / 10000,
                '月内最大亏损额(万元)': max_loss_in_month / 10000 if pd.notna(max_loss_in_month) else np.nan
            })
        
        monthly_df = pd.DataFrame(monthly_results)
        return monthly_df


class ReportGenerator:
    """报告生成器"""
    
    @staticmethod
    def print_performance_report(eval_df, title='策略绩效报告'):
        """
        打印绩效报告
        
        Parameters:
        -----------
        eval_df : pd.DataFrame
            评价结果DataFrame
        title : str
            报告标题
        """
        print("\n" + "=" * 80)
        print(f"{title:^80}")
        print("=" * 80)
        print(eval_df.round(4).to_string())
        print("=" * 80)
    
    @staticmethod
    def print_annual_report(annual_df, title='年度分析报告'):
        """
        打印年度报告
        
        Parameters:
        -----------
        annual_df : pd.DataFrame
            年度分析结果
        title : str
            报告标题
        """
        print("\n" + "=" * 150)
        print(f"{title:^150}")
        print("=" * 150)
        
        for _, row in annual_df.iterrows():
            print(f"{row['年度']:<6} "
                  f"收益: {row['区间收益率']:>7.2%} "
                  f"波动: {row['年化波动率']:>7.2%} "
                  f"回撤: {row['最大回撤']:>7.2%} "
                  f"夏普: {row['夏普比率']:>6.2f} "
                  f"Calmar: {row['Calmar比率']:>6.2f}")
        
        print("=" * 150)
        
        # 计算平均值
        numeric_cols = ['区间收益率', '年化波动率', '最大回撤', 
                       '年初到最低点回撤', '夏普比率', 'Calmar比率']
        
        print("\n整体统计:")
        for col in numeric_cols:
            avg_value = annual_df[col].mean()
            if col in ['区间收益率', '年化波动率', '最大回撤', '年初到最低点回撤']:
                print(f"平均{col}: {avg_value:>7.2%}")
            else:
                print(f"平均{col}: {avg_value:>7.2f}")
        
        print("=" * 150)
    
    @staticmethod
    def save_report(eval_df, annual_df, filepath, filename_prefix='strategy_report'):
        """
        保存报告到Excel
        
        Parameters:
        -----------
        eval_df : pd.DataFrame
            评价结果
        annual_df : pd.DataFrame
            年度分析结果
        filepath : str
            保存路径
        filename_prefix : str
            文件名前缀
        """
        import os
        from datetime import datetime
        
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{filename_prefix}_{timestamp}.xlsx"
        full_path = os.path.join(filepath, filename)
        
        with pd.ExcelWriter(full_path, engine='openpyxl') as writer:
            eval_df.to_excel(writer, sheet_name='绩效评价')
            if annual_df is not None:
                annual_df.to_excel(writer, sheet_name='年度分析', index=False)
        
        print(f"\n报告已保存至: {full_path}")


if __name__ == '__main__':
    # 测试代码
    print("评价模块加载成功!")
    
    # 创建示例数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    returns = pd.Series(np.random.randn(500) * 0.01 + 0.0005, index=dates)
    pv = (returns + 1).cumprod()
    
    # 测试绩效评价
    print("\n测试绩效评价:")
    eval_df = PerformanceEvaluator.evaluate_portfolio(pv, '测试策略')
    ReportGenerator.print_performance_report(eval_df, '测试策略绩效报告')
    
    # 测试年度分析
    print("\n测试年度分析:")
    annual_df = PeriodAnalyzer.annual_analysis(returns)
    print(annual_df.head())

