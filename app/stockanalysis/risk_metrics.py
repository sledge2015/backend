# app/stockanalysis/risk_metrics.py
import pandas as pd
import numpy as np
from typing import Dict, Union, Optional
import logging
from .portfolio_returns import PortfolioReturns

logger = logging.getLogger(__name__)


class RiskMetrics:
	"""风险指标计算模块"""
	
	@staticmethod
	def get_portfolio_sharpe_ratio(
			transactions_df: pd.DataFrame,
			price_df: pd.DataFrame,
			risk_free_rate: Union[float, pd.Series] = 0.03,
			lookback_days: int = 252
			) -> float:
		"""
		计算组合夏普比率（年化）
		"""
		daily_returns = PortfolioReturns.get_portfolio_daily_returns(
			transactions_df, price_df, lookback_days
			)
		
		if len(daily_returns) < 50:
			return 0.0
		
		# 计算年化超额收益和夏普比率
		# 智能处理风险利率类型
		if isinstance(risk_free_rate, pd.Series):
			# 输入是每日风险利率序列
			aligned_risk_free = risk_free_rate.reindex(daily_returns.index, method='ffill').fillna(0)
			excess_return = daily_returns - aligned_risk_free
		elif isinstance(risk_free_rate, (int, float)):
			# 输入是年化风险利率，转换为日利率
			daily_risk_free_rate = risk_free_rate / 252
			excess_return = daily_returns - daily_risk_free_rate
		else:
			raise ValueError("risk_free_rate must be float or pd.Series")
		
		if excess_return.std() == 0:
			return 0.0
		
		sharpe_ratio = (excess_return.mean() / excess_return.std()) * np.sqrt(252)
		return round(sharpe_ratio, 2)
	
	@staticmethod
	def get_portfolio_sortino_ratio(
			transactions_df: pd.DataFrame,
			price_df: pd.DataFrame,
			risk_free_rate: Union[float, pd.Series] = 0.03,
			lookback_days: int = 252
			) -> float:
		"""
		计算组合索提诺比率
		"""
		daily_returns = PortfolioReturns.get_portfolio_daily_returns(
			transactions_df, price_df, lookback_days
			)
		
		if len(daily_returns) < 50:
			return 0.0
		
		# 计算年化收益率
		# 智能处理风险利率类型
		if isinstance(risk_free_rate, pd.Series):
			# 使用每日风险利率计算年化超额收益
			aligned_risk_free = risk_free_rate.reindex(daily_returns.index, method='ffill').fillna(0)
			excess_return = daily_returns - aligned_risk_free
			annual_excess_return = excess_return.mean() * 252
		elif isinstance(risk_free_rate, (int, float)):
			# 使用年化风险利率
			annual_return = daily_returns.mean() * 252
			annual_excess_return = annual_return - risk_free_rate
		else:
			raise ValueError("risk_free_rate must be float or pd.Series")
		
		# 计算下行波动率（只考虑负收益）
		negative_returns = daily_returns[daily_returns < 0]
		
		if len(negative_returns) == 0:
			return 999.99
		
		downside_volatility = negative_returns.std() * np.sqrt(252)
		
		if downside_volatility == 0:
			return 0.0
		
		# 计算索提诺比率
		sortino_ratio = annual_excess_return / downside_volatility
		return round(sortino_ratio, 2)
	
	@staticmethod
	def get_portfolio_max_drawdown(
			transactions_df: pd.DataFrame,
			price_df: pd.DataFrame,
			lookback_days: int = 252
			) -> Dict[str, any]:
		"""
		计算组合最大回撤信息
		"""
		daily_returns = PortfolioReturns.get_portfolio_daily_returns(
			transactions_df, price_df, lookback_days
			)
		
		if len(daily_returns) < 50:
			return {
				'max_drawdown': 0.0,
				'drawdown_start_date': None,
				'drawdown_end_date': None,
				'recovery_date': None,
				'duration_days': 0
				}
		
		# 计算累计收益和回撤
		cumulative = (1 + daily_returns).cumprod()
		rolling_max = cumulative.cummax()
		drawdown = (cumulative - rolling_max) / rolling_max
		
		# 最大回撤值和日期
		max_drawdown = drawdown.min() * 100
		drawdown_end_date = drawdown.idxmin()
		
		# 回撤开始日期（最近的历史高点）
		drawdown_start_date = rolling_max.loc[:drawdown_end_date].idxmax()
		
		# 恢复日期（回撤后第一次创新高的日期）
		recovery_date = None
		peak_value = rolling_max.loc[drawdown_start_date]
		
		for date in cumulative.loc[drawdown_end_date:].index:
			if cumulative.loc[date] >= peak_value:
				recovery_date = date
				break
		
		# 计算持续时间
		if recovery_date:
			duration_days = (recovery_date - drawdown_start_date).days
		else:
			duration_days = (cumulative.index[-1] - drawdown_start_date).days
		
		return {
			'max_drawdown': round(max_drawdown, 2),
			'drawdown_start_date': drawdown_start_date,
			'drawdown_end_date': drawdown_end_date,
			'recovery_date': recovery_date,
			'duration_days': duration_days
			}
	
	@staticmethod
	def get_portfolio_volatility(
			transactions_df: pd.DataFrame,
			price_df: pd.DataFrame,
			lookback_days: int = 252
			) -> float:
		"""
		计算组合年化波动率（按252交易日计算）
		"""
		daily_returns = PortfolioReturns.get_portfolio_daily_returns(
			transactions_df, price_df, lookback_days
			)
		
		if len(daily_returns) < 50:
			return 0.0
		
		volatility = daily_returns.std() * np.sqrt(252) * 100
		return round(volatility, 2)
	
	@staticmethod
	def get_portfolio_annualized_return(
			transactions_df: pd.DataFrame,
			price_df: pd.DataFrame
			) -> float:
		"""
		计算组合年化收益率
		"""
		daily_returns = PortfolioReturns.get_portfolio_daily_returns(
			transactions_df, price_df, lookback_days=None
			)
		
		daily_returns = daily_returns.dropna()  # 去除NaN值
		if len(daily_returns) < 50:
			return 0.0
		
		# 计算总收益率
		total_return = (1 + daily_returns).prod() - 1
		
		# 计算年化收益率
		trading_days = len(daily_returns)
		annualized_return = (1 + total_return) ** (252 / trading_days) - 1
		
		return round(annualized_return * 100, 2)
	
	@staticmethod
	def calculate_concentration_metrics(daily_positions: pd.DataFrame) -> Dict[str, float]:
		"""
		计算投资组合集中度指标
		"""
		if daily_positions.empty:
			return {'herfindahl_index': 0.0, 'effective_number_stocks': 0.0}
		
		# 获取最新日期的持仓权重
		latest_date = daily_positions['datetime'].max()
		latest_positions = daily_positions[daily_positions['datetime'] == latest_date]
		
		if latest_positions.empty:
			return {'herfindahl_index': 0.0, 'effective_number_stocks': 0.0}
		
		# 过滤掉零持仓
		active_positions = latest_positions[latest_positions['quantity'] != 0]
		
		if active_positions.empty:
			return {'herfindahl_index': 0.0, 'effective_number_stocks': 0.0}
		
		# 计算权重（确保权重和为1）
		weights = active_positions['weight'].values
		weights = weights / weights.sum() if weights.sum() > 0 else weights
		
		# 赫芬达尔指数 = Σ(wi²)
		herfindahl_index = np.sum(weights ** 2)
		
		# 有效股票数量 = 1 / HHI
		effective_number_stocks = 1.0 / herfindahl_index if herfindahl_index > 0 else 0.0
		
		return {
			'herfindahl_index': round(herfindahl_index, 4),
			'effective_number_stocks': round(effective_number_stocks, 2)
			}