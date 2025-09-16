# backend/app/services/stock_metrics.py
import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay
from decimal import Decimal
from typing import List, Dict, Optional, Union
import numpy as np
import pytz

import logging

logger = logging.getLogger(__name__)


class StockMetrics:
	"""股票分析服务类（高效 Pandas 版本）"""
	
	@staticmethod
	def get_historical_returns(
			price_df: pd.DataFrame,
			symbols: List[str],
			periods: List[str] = ['1W', 'YTD']
			) -> Dict[str, Dict[str, float]]:
		"""
        获取股票历史收益率（按自然周期计算），基于 Pandas。

        price_df: DataFrame，包含 columns=['stock_symbol', 'datetime', 'close']，datetime 必须是 pd.Timestamp
        symbols: 股票代码列表
        periods: 时间周期列表，支持 '1D','1W','2W','1M','3M','6M','1Y','YTD','MTD'
        返回: { "AAPL": {"1D": 1.23, "1W": 5.67, "1M": 10.12}, ... }
        """
		if not symbols or price_df.empty:
			return {}
		
		symbols = [s.upper() for s in symbols]
		price_df = price_df[price_df['stock_symbol'].isin(symbols)].copy()
		price_df['datetime'] = pd.to_datetime(price_df['datetime']).dt.normalize()
		price_df.sort_values(['stock_symbol', 'datetime'], inplace=True)
		price_df.set_index('datetime', inplace=True)
		
		# 所有交易日
		trading_days = price_df.index.unique().sort_values()
		result: Dict[str, Dict[str, float]] = {s: {} for s in symbols}
		
		# 使用数据中的最新日期作为基准
		latest_date = trading_days.max()
		print(f"使用最新数据日期作为基准: {latest_date}")
		
		# 最近收盘价
		latest_prices = price_df.groupby('stock_symbol')['close'].last()
		
		# 自定义交易日偏移
		all_weekdays = pd.bdate_range(trading_days.min(), trading_days.max())
		holidays = all_weekdays.difference(trading_days)
		us_bday = CustomBusinessDay(holidays=holidays)
		
		# 计算目标日期函数
		def calc_target_date(current: pd.Timestamp, period: str) -> pd.Timestamp:
			if period == '1D':
				return current - us_bday
			elif period == '1W':
				return current - 5 * us_bday
			elif period == '2W':
				return current - 10 * us_bday
			elif period == '1M':
				return StockMetrics._adjust_to_trading_day(current - pd.DateOffset(months=1), trading_days)
			elif period == '3M':
				return StockMetrics._adjust_to_trading_day(current - pd.DateOffset(months=3), trading_days)
			elif period == '6M':
				return StockMetrics._adjust_to_trading_day(current - pd.DateOffset(months=6), trading_days)
			elif period == '1Y':
				return StockMetrics._adjust_to_trading_day(current - pd.DateOffset(years=1), trading_days)
			elif period == 'YTD':
				return StockMetrics._adjust_to_trading_day(current.replace(month=1, day=1), trading_days)
			elif period == 'MTD':
				return StockMetrics._adjust_to_trading_day(current.replace(day=1), trading_days)
			else:
				return None
		
		# 遍历每只股票
		for symbol in symbols:
			latest_price = latest_prices.get(symbol, None)
			if pd.isna(latest_price) or latest_price == 0:
				continue
			for period in periods:
				target_date = calc_target_date(latest_date, period)
				if target_date is None:
					result[symbol][period] = 0.0
					continue
				# 查找最近交易日价格
				past_prices = price_df.loc[
					(price_df['stock_symbol'] == symbol) & (price_df.index <= target_date), 'close']
				if past_prices.empty:
					result[symbol][period] = 0.0
					continue
				past_price = past_prices.iloc[-1]
				if past_price == 0:
					result[symbol][period] = 0.0
					continue
				returns = (Decimal(str(latest_price)) - Decimal(str(past_price))) / Decimal(str(past_price)) * Decimal(
					'100')
				result[symbol][period] = float(returns.quantize(Decimal('0.01')))
		return result
	
	@staticmethod
	def get_portfolio_metrics(
			transactions_df: pd.DataFrame,
			price_df: pd.DataFrame,
			benchmark_returns: Optional[pd.Series] = None,
			risk_free_rate: Union[float, pd.Series] = 0.03,
			lookback_days: int = 252
			) -> Dict[str, any]:
		"""
		获取投资组合完整风险指标（带日志输出）
		"""
		logger.info("开始计算投资组合风险指标...")
		
		# 获取每日收益率
		daily_returns = StockMetrics._get_portfolio_daily_returns(
			transactions_df, price_df, lookback_days
			)
		logger.debug("每日收益率计算完成，样本数=%d", len(daily_returns))
		
		# 获取每日持仓数据
		daily_positions = StockMetrics._build_daily_positions(
			transactions_df, price_df
			)
		logger.debug("每日持仓构建完成，样本数=%d", len(daily_positions))
		
		if daily_returns.empty or daily_positions.empty:
			logger.warning("数据为空，返回空指标结果")
			return StockMetrics._get_empty_metrics()
		
		# 1. 夏普比率
		sharpe_ratio = StockMetrics.get_portfolio_sharpe_ratio(
			transactions_df, price_df, risk_free_rate, lookback_days
			)
		logger.info("夏普比率=%.4f", sharpe_ratio)
		
		# 2. 索提诺比率
		sortino_ratio = StockMetrics.get_portfolio_sortino_ratio(
			transactions_df, price_df, risk_free_rate, lookback_days
			)
		logger.info("索提诺比率=%.4f", sortino_ratio)
		
		# 3. Beta系数
		beta = 0.0
		if benchmark_returns is not None and not benchmark_returns.empty:
			aligned_data = pd.DataFrame({
				'portfolio': daily_returns,
				'benchmark': benchmark_returns
				}).dropna()
			
			if len(aligned_data) >= 50:
				covariance = np.cov(aligned_data['portfolio'], aligned_data['benchmark'])[0, 1]
				benchmark_variance = np.var(aligned_data['benchmark'])
				if benchmark_variance > 0:
					beta = round(covariance / benchmark_variance, 3)
		logger.info("Beta系数=%.4f", beta)
		
		# 4. 最大回撤
		drawdown_info = StockMetrics.get_portfolio_max_drawdown(
			transactions_df, price_df, lookback_days
			)
		logger.info("最大回撤=%.2f%%, 持续=%d天",
		            drawdown_info['max_drawdown'], drawdown_info['duration_days'])
		
		# 5. 波动率
		volatility = StockMetrics.get_portfolio_volatility(
			transactions_df, price_df, lookback_days
			)
		logger.info("年化波动率=%.2f%%", volatility)
		
		
		metrics = {
			'sharpe_ratio': sharpe_ratio,
			'sortino_ratio': sortino_ratio,
			'beta': beta,
			'max_drawdown': drawdown_info['max_drawdown'],
			'drawdown_start_date': drawdown_info['drawdown_start_date'],
			'drawdown_end_date': drawdown_info['drawdown_end_date'],
			'recovery_date': drawdown_info['recovery_date'],
			'drawdown_duration_days': drawdown_info['duration_days'],
			'volatility': volatility,
			'total_trading_days': len(daily_returns),
			'latest_update_date': daily_returns.index[-1] if not daily_returns.empty else None
			}
		
		logger.info("风险指标计算完成，交易日数=%d, 最新数据日期=%s",
		            metrics['total_trading_days'], metrics['latest_update_date'])
		
		return metrics
	
	@staticmethod
	def _get_empty_metrics() -> Dict[str, any]:
		"""返回空的指标字典"""
		return {
			'sharpe_ratio': 0.0,
			'sortino_ratio': 0.0,
			'beta': 0.0,
			'max_drawdown': 0.0,
			'drawdown_start_date': None,
			'drawdown_end_date': None,
			'recovery_date': None,
			'drawdown_duration_days': 0,
			'volatility': 0.0,
			'herfindahl_index': 0.0,
			'effective_number_stocks': 0.0,
			'annualized_return': 0.0,
			'total_return': 0.0,
			'total_trading_days': 0,
			'latest_update_date': None
			}
	
	@staticmethod
	def _adjust_to_trading_day(date: pd.Timestamp, trading_days: pd.DatetimeIndex) -> pd.Timestamp:
		"""调整日期到最近交易日（向前取交易日）"""
		if date in trading_days:
			return date
		prior_days = trading_days[trading_days <= date]
		return prior_days[-1] if len(prior_days) > 0 else None
	
	@staticmethod
	def _build_daily_positions(
			transactions_df: pd.DataFrame,
			price_df: pd.DataFrame
			) -> pd.DataFrame:
		"""
        构建每日持仓表和权重表（向量化）

        Parameters:
        transactions_df: columns=['trade_time', 'stock_symbol', 'quantity', 'price']
        price_df: columns=['datetime', 'stock_symbol', 'close']

        Returns:
        每日持仓DataFrame
        """
		if transactions_df.empty or price_df.empty:
			return pd.DataFrame()
		
		transactions_df = transactions_df.copy()
		price_df = price_df.copy()
		
		# 预处理时间字段
		transactions_df['trade_time'] = pd.to_datetime(transactions_df['trade_time'])
		price_df['datetime'] = pd.to_datetime(price_df['datetime'])
		
		# 1️⃣ 累计交易数量
		transactions_df['trade_date'] = transactions_df['trade_time'].dt.floor('D')
		daily_qty = (
			transactions_df
			.groupby(['trade_date', 'stock_symbol'])['quantity']
			.sum()
			.reset_index()
		)
		
		if daily_qty.empty:
			return pd.DataFrame()
		
		# 2️⃣ 创建完整日期×股票表
		all_dates = pd.date_range(
			daily_qty['trade_date'].min(),
			daily_qty['trade_date'].max(),
			freq='D'
			)
		stocks = daily_qty['stock_symbol'].unique()
		full_index = pd.MultiIndex.from_product(
			[all_dates, stocks],
			names=['datetime', 'stock_symbol']
			)
		daily_positions = daily_qty.set_index(['trade_date', 'stock_symbol']).reindex(
			full_index, fill_value=0
			)
		
		# 3️⃣ 累计持仓
		daily_positions['quantity'] = daily_positions['quantity'].groupby(level=1).cumsum()
		daily_positions = daily_positions.reset_index()
		
		# 4️⃣ 对齐价格（merge_asof）
		price_df_sorted = price_df.sort_values('datetime')
		daily_positions_sorted = daily_positions.sort_values('datetime')
		daily_positions_merged = pd.merge_asof(
			daily_positions_sorted,
			price_df_sorted,
			by='stock_symbol',
			left_on='datetime',
			right_on='datetime',
			direction='backward'
			)
		
		# 过滤掉没有价格的数据
		daily_positions_merged = daily_positions_merged.dropna(subset=['close'])
		
		if daily_positions_merged.empty:
			return pd.DataFrame()
		
		# 5️⃣ 计算每日市值
		daily_positions_merged['market_value'] = (
				daily_positions_merged['quantity'] * daily_positions_merged['close']
		)
		
		# 6️⃣ 计算每日总市值与权重
		daily_total_value = (
			daily_positions_merged
			.groupby('datetime')['market_value']
			.sum()
			.rename('total_value')
		)
		daily_positions_merged = daily_positions_merged.merge(daily_total_value, on='datetime')
		
		# 避免除零错误
		daily_positions_merged['weight'] = np.where(
			daily_positions_merged['total_value'] > 0,
			daily_positions_merged['market_value'] / daily_positions_merged['total_value'],
			0
			)
		
		return daily_positions_merged
	
	@staticmethod
	def _get_portfolio_daily_returns(
			transactions_df: pd.DataFrame,
			price_df: pd.DataFrame,
			lookback_days: Optional[int] = None
			) -> pd.Series:
		"""
		正确计算每日组合收益率 - 基于持仓权重
		"""
		daily_positions = StockMetrics._build_daily_positions(
		    transactions_df, price_df
		)
		
		if daily_positions.empty:
		    return pd.Series()
		
		# 按日期分组
		daily_returns = []
		dates = sorted(daily_positions['datetime'].unique())
		
		for i, date in enumerate(dates):
		    if i == 0:
		        continue  # 第一天没有收益率
		        
		    prev_date = dates[i-1]
		    
		    # 获取前一日持仓
		    prev_positions = daily_positions[daily_positions['datetime'] == prev_date]
		    curr_positions = daily_positions[daily_positions['datetime'] == date]
		    
		    # 计算组合收益率 = Σ(权重i × 股票i收益率)
		    portfolio_return = 0
		    prev_total_value = prev_positions['market_value'].sum()
		    
		    for _, prev_pos in prev_positions.iterrows():
		        symbol = prev_pos['stock_symbol']
		        prev_value = prev_pos['market_value']
		        weight = prev_value / prev_total_value
		        
		        # 计算该股票的收益率
		        prev_price = prev_pos['close']
		        curr_price = price_df.loc[date, symbol] if date in price_df.index else prev_price
		        stock_return = (curr_price - prev_price) / prev_price
		        
		        portfolio_return += weight * stock_return
		    
		    daily_returns.append(portfolio_return)
		
		return pd.Series(daily_returns, index=dates[1:])
	
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
		daily_returns = StockMetrics._get_portfolio_daily_returns(
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
	def get_portfolio_max_drawdown(
			transactions_df: pd.DataFrame,
			price_df: pd.DataFrame,
			lookback_days: int = 252
			) -> Dict[str, any]:
		"""
        计算组合最大回撤信息
        """
		daily_returns = StockMetrics._get_portfolio_daily_returns(
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
		daily_returns = StockMetrics._get_portfolio_daily_returns(
			transactions_df, price_df, lookback_days
			)
		
		if len(daily_returns) < 50:
			return 0.0
		
		volatility = daily_returns.std() * np.sqrt(252) * 100
		return round(volatility, 2)
	
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
		daily_returns = StockMetrics._get_portfolio_daily_returns(
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
	def get_daily_positions(
			transactions_df: pd.DataFrame,
			price_df: pd.DataFrame,
			lookback_days: Optional[int] = None
			) -> pd.DataFrame:
		"""
        获取每日持仓和权重表
        """
		daily_positions = StockMetrics._build_daily_positions(
			transactions_df, price_df
			)
		
		if lookback_days and not daily_positions.empty:
			latest_date = daily_positions['datetime'].max()
			start_date = latest_date - pd.Timedelta(days=lookback_days)
			daily_positions = daily_positions[daily_positions['datetime'] >= start_date]
		
		return daily_positions
