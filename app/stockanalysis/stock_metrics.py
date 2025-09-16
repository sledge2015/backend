# backend/app/services/stock_metrics.py
import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay
from decimal import Decimal
from typing import List, Dict, Optional, Union
import numpy as np
import pytz
import logging

logger = logging.getLogger(__name__)

# 检查QuantStats是否可用
try:
	import quantstats as qs
	
	HAS_QUANTSTATS = True
	logger.info("✅ QuantStats库可用，将使用专业算法计算风险指标")
except ImportError:
	HAS_QUANTSTATS = False
	logger.warning("❌ QuantStats库不可用，将使用备用算法。建议安装: pip install quantstats")

# 备用库检查
try:
	import empyrical as emp
	
	HAS_EMPYRICAL = True
	logger.info("✅ Empyrical库可用作为备用")
except ImportError:
	HAS_EMPYRICAL = False
	logger.warning("❌ Empyrical库不可用，建议安装: pip install empyrical")


class StockMetrics:
	"""股票分析服务类（使用QuantStats优化版本）"""
	
	@staticmethod
	def get_historical_returns(
			price_df: pd.DataFrame,
			symbols: List[str],
			periods: List[str] = ['1W', 'YTD']
			) -> Dict[str, Dict[str, float]]:
		"""
		获取股票历史收益率（按自然周期计算），基于 Pandas。
		保持原有逻辑不变。
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
		logger.debug(f"使用最新数据日期作为基准: {latest_date}")
		
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
		获取投资组合完整风险指标（使用QuantStats优化）
		"""
		logger.info("开始计算投资组合风险指标（QuantStats优化版本）...")
		
		# 获取每日收益率
		daily_returns = StockMetrics._get_portfolio_daily_returns(
			transactions_df, price_df, lookback_days
			)
		logger.debug("每日收益率计算完成，样本数=%d", len(daily_returns))
		
		if daily_returns.empty:
			logger.warning("数据为空，返回空指标结果")
			return StockMetrics._get_empty_metrics()
		
		# 确保daily_returns是pandas Series且有DatetimeIndex
		if not isinstance(daily_returns.index, pd.DatetimeIndex):
			daily_returns.index = pd.to_datetime(daily_returns.index)
		
		# 使用QuantStats计算各项指标
		sharpe_ratio = StockMetrics.get_portfolio_sharpe_ratio_qs(daily_returns, risk_free_rate)
		sortino_ratio = StockMetrics.get_portfolio_sortino_ratio_qs(daily_returns, risk_free_rate)
		max_drawdown_info = StockMetrics.get_portfolio_max_drawdown_qs(daily_returns)
		volatility = StockMetrics.get_portfolio_volatility_qs(daily_returns)
		beta = StockMetrics.get_portfolio_beta_qs(daily_returns, benchmark_returns)
		
		# 额外的QuantStats指标
		var_95 = StockMetrics.get_portfolio_var_qs(daily_returns)
		cvar_95 = StockMetrics.get_portfolio_cvar_qs(daily_returns)
		calmar_ratio = StockMetrics.get_portfolio_calmar_ratio_qs(daily_returns)
		
		logger.info("夏普比率=%.4f", sharpe_ratio)
		logger.info("索提诺比率=%.4f", sortino_ratio)
		logger.info("Beta系数=%.4f", beta)
		logger.info("最大回撤=%.2f%%", max_drawdown_info['max_drawdown'])
		logger.info("年化波动率=%.2f%%", volatility)
		
		metrics = {
			'sharpe_ratio': sharpe_ratio,
			'sortino_ratio': sortino_ratio,
			'beta': beta,
			'max_drawdown': max_drawdown_info['max_drawdown'],
			'drawdown_start_date': max_drawdown_info.get('drawdown_start_date'),
			'drawdown_end_date': max_drawdown_info.get('drawdown_end_date'),
			'recovery_date': max_drawdown_info.get('recovery_date'),
			'drawdown_duration_days': max_drawdown_info.get('duration_days', 0),
			'volatility': volatility,
			'var_95': var_95,
			'cvar_95': cvar_95,
			'calmar_ratio': calmar_ratio,
			'total_trading_days': len(daily_returns),
			'latest_update_date': daily_returns.index[-1] if not daily_returns.empty else None,
			'calculation_method': 'QuantStats' if HAS_QUANTSTATS else 'Manual'
			}
		
		logger.info("风险指标计算完成，交易日数=%d, 最新数据日期=%s, 使用方法=%s",
		            metrics['total_trading_days'], metrics['latest_update_date'], metrics['calculation_method'])
		
		return metrics
	
	@staticmethod
	def get_portfolio_sharpe_ratio_qs(
			daily_returns: pd.Series,
			risk_free_rate: Union[float, pd.Series] = 0.03
			) -> float:
		"""使用QuantStats计算夏普比率"""
		if len(daily_returns) < 50:
			return 0.0
		
		try:
			if HAS_QUANTSTATS:
				result = qs.stats.sharpe(daily_returns, rf=risk_free_rate)
				return round(float(result), 4)
			elif HAS_EMPYRICAL:
				result = emp.sharpe_ratio(daily_returns, risk_free=risk_free_rate)
				return round(float(result), 4)
			else:
				# 备用手工计算
				return StockMetrics._manual_sharpe_ratio(daily_returns, risk_free_rate)
		except Exception as e:
			logger.warning(f"QuantStats夏普比率计算失败: {e}，使用备用方法")
			return StockMetrics._manual_sharpe_ratio(daily_returns, risk_free_rate)
	
	@staticmethod
	def get_portfolio_sortino_ratio_qs(
			daily_returns: pd.Series,
			risk_free_rate: Union[float, pd.Series] = 0.03
			) -> float:
		"""使用QuantStats计算索提诺比率"""
		if len(daily_returns) < 50:
			return 0.0
		
		try:
			if HAS_QUANTSTATS:
				result = qs.stats.sortino(daily_returns, rf=risk_free_rate)
				return round(float(result), 4)
			elif HAS_EMPYRICAL:
				result = emp.sortino_ratio(daily_returns, required_return=risk_free_rate)
				return round(float(result), 4)
			else:
				return StockMetrics._manual_sortino_ratio(daily_returns, risk_free_rate)
		except Exception as e:
			logger.warning(f"QuantStats索提诺比率计算失败: {e}，使用备用方法")
			return StockMetrics._manual_sortino_ratio(daily_returns, risk_free_rate)
	
	@staticmethod
	def get_portfolio_max_drawdown_qs(daily_returns: pd.Series) -> Dict[str, any]:
		"""使用QuantStats计算最大回撤"""
		if len(daily_returns) < 50:
			return {
				'max_drawdown': 0.0,
				'drawdown_start_date': None,
				'drawdown_end_date': None,
				'recovery_date': None,
				'duration_days': 0
				}
		
		try:
			if HAS_QUANTSTATS:
				# QuantStats计算最大回撤
				max_dd = qs.stats.max_drawdown(daily_returns)
				max_drawdown_value = round(float(max_dd) * 100, 2)
				
				# 计算详细的回撤信息
				drawdown_details = StockMetrics._calculate_drawdown_details_qs(daily_returns)
				drawdown_details['max_drawdown'] = max_drawdown_value
				
				return drawdown_details
			elif HAS_EMPYRICAL:
				max_dd = emp.max_drawdown(daily_returns)
				max_drawdown_value = round(float(max_dd) * 100, 2)
				drawdown_details = StockMetrics._calculate_drawdown_details_qs(daily_returns)
				drawdown_details['max_drawdown'] = max_drawdown_value
				return drawdown_details
			else:
				return StockMetrics._manual_max_drawdown(daily_returns)
		except Exception as e:
			logger.warning(f"QuantStats最大回撤计算失败: {e}，使用备用方法")
			return StockMetrics._manual_max_drawdown(daily_returns)
	
	@staticmethod
	def get_portfolio_volatility_qs(daily_returns: pd.Series) -> float:
		"""使用QuantStats计算波动率"""
		if len(daily_returns) < 50:
			return 0.0
		
		try:
			if HAS_QUANTSTATS:
				result = qs.stats.volatility(daily_returns)
				return round(float(result) * 100, 2)
			elif HAS_EMPYRICAL:
				result = emp.annual_volatility(daily_returns)
				return round(float(result) * 100, 2)
			else:
				return round(daily_returns.std() * np.sqrt(252) * 100, 2)
		except Exception as e:
			logger.warning(f"QuantStats波动率计算失败: {e}，使用备用方法")
			return round(daily_returns.std() * np.sqrt(252) * 100, 2)
	
	@staticmethod
	def get_portfolio_beta_qs(
			daily_returns: pd.Series,
			benchmark_returns: Optional[pd.Series] = None
			) -> float:
		"""使用QuantStats计算Beta系数"""
		if benchmark_returns is None or benchmark_returns.empty or len(daily_returns) < 50:
			return 0.0
		
		try:
			# 对齐数据
			aligned_data = pd.DataFrame({
				'portfolio': daily_returns,
				'benchmark': benchmark_returns
				}).dropna()
			
			if len(aligned_data) < 50:
				return 0.0
			
			if HAS_EMPYRICAL:
				result = emp.beta(aligned_data['portfolio'], aligned_data['benchmark'])
				return round(float(result), 4)
			else:
				# 手工计算Beta
				covariance = np.cov(aligned_data['portfolio'], aligned_data['benchmark'])[0, 1]
				benchmark_variance = np.var(aligned_data['benchmark'])
				return round(covariance / benchmark_variance if benchmark_variance > 0 else 0.0, 4)
		except Exception as e:
			logger.warning(f"Beta计算失败: {e}")
			return 0.0
	
	@staticmethod
	def get_portfolio_var_qs(daily_returns: pd.Series, confidence: float = 0.05) -> float:
		"""使用QuantStats计算VaR（Value at Risk）"""
		if len(daily_returns) < 50:
			return 0.0
		
		try:
			if HAS_QUANTSTATS:
				result = qs.stats.var(daily_returns, confidence=confidence)
				return round(float(result) * 100, 2)
			else:
				# 手工计算VaR
				return round(float(daily_returns.quantile(confidence)) * 100, 2)
		except Exception as e:
			logger.warning(f"VaR计算失败: {e}")
			return 0.0
	
	@staticmethod
	def get_portfolio_cvar_qs(daily_returns: pd.Series, confidence: float = 0.05) -> float:
		"""使用QuantStats计算CVaR（Conditional Value at Risk）"""
		if len(daily_returns) < 50:
			return 0.0
		
		try:
			if HAS_QUANTSTATS:
				result = qs.stats.cvar(daily_returns, confidence=confidence)
				return round(float(result) * 100, 2)
			else:
				# 手工计算CVaR
				var_threshold = daily_returns.quantile(confidence)
				cvar = daily_returns[daily_returns <= var_threshold].mean()
				return round(float(cvar) * 100, 2)
		except Exception as e:
			logger.warning(f"CVaR计算失败: {e}")
			return 0.0
	
	@staticmethod
	def get_portfolio_calmar_ratio_qs(daily_returns: pd.Series) -> float:
		"""使用QuantStats计算卡尔玛比率"""
		if len(daily_returns) < 50:
			return 0.0
		
		try:
			if HAS_QUANTSTATS:
				result = qs.stats.calmar(daily_returns)
				return round(float(result), 4)
			else:
				# 手工计算：年化收益率 / 最大回撤
				annual_return = (1 + daily_returns).cumprod().iloc[-1] ** (252 / len(daily_returns)) - 1
				cumulative = (1 + daily_returns).cumprod()
				max_dd = ((cumulative.cummax() - cumulative) / cumulative.cummax()).max()
				return round(annual_return / max_dd if max_dd > 0 else 0.0, 4)
		except Exception as e:
			logger.warning(f"卡尔玛比率计算失败: {e}")
			return 0.0
	
	@staticmethod
	def _calculate_drawdown_details_qs(daily_returns: pd.Series) -> Dict[str, any]:
		"""计算详细的回撤信息"""
		try:
			# 计算累计收益和回撤
			cumulative = (1 + daily_returns).cumprod()
			rolling_max = cumulative.cummax()
			drawdown = (cumulative - rolling_max) / rolling_max
			
			# 最大回撤日期
			drawdown_end_date = drawdown.idxmin()
			
			# 回撤开始日期
			drawdown_start_date = rolling_max.loc[:drawdown_end_date].idxmax()
			
			# 恢复日期
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
				'drawdown_start_date': drawdown_start_date,
				'drawdown_end_date': drawdown_end_date,
				'recovery_date': recovery_date,
				'duration_days': duration_days
				}
		except Exception as e:
			logger.warning(f"回撤详情计算失败: {e}")
			return {
				'drawdown_start_date': None,
				'drawdown_end_date': None,
				'recovery_date': None,
				'duration_days': 0
				}
	
	# ===== 备用手工计算方法 =====
	@staticmethod
	def _manual_sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float) -> float:
		"""备用的手工夏普比率计算"""
		try:
			daily_rf = risk_free_rate / 252
			excess_returns = daily_returns - daily_rf
			if excess_returns.std() == 0:
				return 0.0
			sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
			return round(sharpe, 4)
		except:
			return 0.0
	
	@staticmethod
	def _manual_sortino_ratio(daily_returns: pd.Series, risk_free_rate: float) -> float:
		"""备用的手工索提诺比率计算"""
		try:
			annual_return = daily_returns.mean() * 252
			annual_excess_return = annual_return - risk_free_rate
			negative_returns = daily_returns[daily_returns < 0]
			if len(negative_returns) == 0:
				return 999.99
			downside_volatility = negative_returns.std() * np.sqrt(252)
			if downside_volatility == 0:
				return 0.0
			sortino = annual_excess_return / downside_volatility
			return round(sortino, 4)
		except:
			return 0.0
	
	@staticmethod
	def _manual_max_drawdown(daily_returns: pd.Series) -> Dict[str, any]:
		"""备用的手工最大回撤计算"""
		try:
			cumulative = (1 + daily_returns).cumprod()
			rolling_max = cumulative.cummax()
			drawdown = (cumulative - rolling_max) / rolling_max
			max_drawdown = drawdown.min() * 100
			
			return {
				'max_drawdown': round(max_drawdown, 2),
				'drawdown_start_date': None,
				'drawdown_end_date': None,
				'recovery_date': None,
				'duration_days': 0
				}
		except:
			return {
				'max_drawdown': 0.0,
				'drawdown_start_date': None,
				'drawdown_end_date': None,
				'recovery_date': None,
				'duration_days': 0
				}
	
	# ===== 保持原有的辅助方法 =====
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
			'var_95': 0.0,
			'cvar_95': 0.0,
			'calmar_ratio': 0.0,
			'total_trading_days': 0,
			'latest_update_date': None,
			'calculation_method': 'Empty'
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
		"""构建每日持仓表和权重表（保持原有逻辑）"""
		if transactions_df.empty or price_df.empty:
			return pd.DataFrame()
		
		transactions_df = transactions_df.copy()
		price_df = price_df.copy()
		
		# 预处理时间字段
		transactions_df['trade_time'] = pd.to_datetime(transactions_df['trade_time'])
		price_df['datetime'] = pd.to_datetime(price_df['datetime'])
		
		# 累计交易数量
		transactions_df['trade_date'] = transactions_df['trade_time'].dt.floor('D')
		daily_qty = (
			transactions_df
			.groupby(['trade_date', 'stock_symbol'])['quantity']
			.sum()
			.reset_index()
		)
		
		if daily_qty.empty:
			return pd.DataFrame()
		
		# 创建完整日期×股票表
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
		
		# 累计持仓
		daily_positions['quantity'] = daily_positions['quantity'].groupby(level=1).cumsum()
		daily_positions = daily_positions.reset_index()
		
		# 对齐价格
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
		
		# 计算每日市值
		daily_positions_merged['market_value'] = (
				daily_positions_merged['quantity'] * daily_positions_merged['close']
		)
		
		# 计算每日总市值与权重
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
		计算每日组合收益率 - 修复版本（使用向量化操作）
		"""
		daily_positions = StockMetrics._build_daily_positions(transactions_df, price_df)
		
		if daily_positions.empty:
			return pd.Series()
		
		# 计算每日组合总价值
		daily_portfolio_value = (
			daily_positions
			.groupby('datetime')['market_value']
			.sum()
			.sort_index()
		)
		
		# 计算收益率
		portfolio_returns = daily_portfolio_value.pct_change().dropna()
		
		# 应用lookback限制
		if lookback_days and not portfolio_returns.empty:
			cutoff_date = portfolio_returns.index[-1] - pd.Timedelta(days=lookback_days)
			portfolio_returns = portfolio_returns[portfolio_returns.index >= cutoff_date]
		
		return portfolio_returns
	
	# ===== 保持兼容性的旧方法名 =====
	@staticmethod
	def get_portfolio_sharpe_ratio(
			transactions_df: pd.DataFrame,
			price_df: pd.DataFrame,
			risk_free_rate: Union[float, pd.Series] = 0.03,
			lookback_days: int = 252
			) -> float:
		"""保持兼容性的夏普比率计算（内部调用QuantStats版本）"""
		daily_returns = StockMetrics._get_portfolio_daily_returns(transactions_df, price_df, lookback_days)
		return StockMetrics.get_portfolio_sharpe_ratio_qs(daily_returns, risk_free_rate)
	
	@staticmethod
	def get_portfolio_sortino_ratio(
			transactions_df: pd.DataFrame,
			price_df: pd.DataFrame,
			risk_free_rate: Union[float, pd.Series] = 0.03,
			lookback_days: int = 252
			) -> float:
		"""保持兼容性的索提诺比率计算"""
		daily_returns = StockMetrics._get_portfolio_daily_returns(transactions_df, price_df, lookback_days)
		return StockMetrics.get_portfolio_sortino_ratio_qs(daily_returns, risk_free_rate)
	
	@staticmethod
	def get_portfolio_max_drawdown(
			transactions_df: pd.DataFrame,
			price_df: pd.DataFrame,
			lookback_days: int = 252
			) -> Dict[str, any]:
		"""保持兼容性的最大回撤计算"""
		daily_returns = StockMetrics._get_portfolio_daily_returns(transactions_df, price_df, lookback_days)
		return StockMetrics.get_portfolio_max_drawdown_qs(daily_returns)
	
	@staticmethod
	def get_portfolio_volatility(
			transactions_df: pd.DataFrame,
			price_df: pd.DataFrame,
			lookback_days: int = 252
			) -> float:
		"""保持兼容性的波动率计算"""
		daily_returns = StockMetrics._get_portfolio_daily_returns(transactions_df, price_df, lookback_days)
		return StockMetrics.get_portfolio_volatility_qs(daily_returns)
	
	@staticmethod
	def get_daily_positions(
			transactions_df: pd.DataFrame,
			price_df: pd.DataFrame,
			lookback_days: Optional[int] = None
			) -> pd.DataFrame:
		"""获取每日持仓和权重表"""
		daily_positions = StockMetrics._build_daily_positions(transactions_df, price_df)
		
		if lookback_days and not daily_positions.empty:
			latest_date = daily_positions['datetime'].max()
			start_date = latest_date - pd.Timedelta(days=lookback_days)
			daily_positions = daily_positions[daily_positions['datetime'] >= start_date]
		
		return daily_positions
