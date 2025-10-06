import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from data.stock_data import StockDataProvider
from models.ml_stock_predictor import MLStockPredictor

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """取引記録"""

    symbol: str
    entry_date: datetime
    exit_date: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    trade_type: str  # 'buy' or 'sell'
    score: float
    return_pct: Optional[float] = None
    holding_days: Optional[int] = None


@dataclass
class BacktestResult:
    """バックテスト結果"""

    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    total_trades: int
    avg_holding_days: float
    best_trade: float
    worst_trade: float
    trades: List[Trade]


class Backtester:
    """バックテストエンジン"""

    def __init__(self, initial_capital: float = 1000000):
        self.initial_capital = initial_capital
        self.data_provider = StockDataProvider()

    def run_backtest(
        self,
        predictor,
        symbols: List[str],
        start_date: str,
        end_date: str,
        rebalance_frequency: int = 5,  # 日数
        top_n: int = 3,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.10,
        max_holding_days: int = 30,
        score_threshold: float = 60,  # 新しいパラメータ
    ) -> BacktestResult:
        """バックテストを実行"""
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        logger.info(f"Score threshold: {score_threshold}")

        # 全銘柄のデータを取得
        all_data = {}
        for symbol in symbols:
            try:
                data = self.data_provider.get_stock_data(symbol, "3y")
                if not data.empty:
                    all_data[symbol] = data
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {e!s}")
                continue

        if not all_data:
            raise ValueError("No valid stock data available")

        # 日付範囲を設定
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # ポートフォリオ状態
        cash = self.initial_capital
        positions = {}  # symbol -> quantity
        trades = []
        portfolio_values = []

        # 取引日を生成
        trading_days = pd.bdate_range(start=start_dt, end=end_dt)

        for current_date in trading_days:
            try:
                # 現在のポートフォリオ価値を計算
                portfolio_value = cash
                for symbol, quantity in positions.items():
                    if symbol in all_data:
                        current_price = self._get_price_on_date(
                            all_data[symbol], current_date,
                        )
                        if current_price is not None:
                            portfolio_value += quantity * current_price

                portfolio_values.append(
                    {
                        "date": current_date,
                        "portfolio_value": portfolio_value,
                        "cash": cash,
                    },
                )

                # 既存ポジションの管理（損切り、利確、保有期間チェック）
                trades_to_close = []
                for trade in trades:
                    if trade.exit_date is None and trade.symbol in all_data:
                        current_price = self._get_price_on_date(
                            all_data[trade.symbol], current_date,
                        )
                        if current_price is None:
                            continue

                        # 保有日数チェック
                        holding_days = (
                            current_date.date() - trade.entry_date.date()
                        ).days

                        # 利確・損切り判定
                        if trade.trade_type == "buy":
                            return_pct = (
                                current_price - trade.entry_price
                            ) / trade.entry_price

                            should_close = (
                                return_pct <= -stop_loss_pct  # 損切り
                                or return_pct >= take_profit_pct  # 利確
                                or holding_days >= max_holding_days  # 最大保有期間
                            )

                            if should_close:
                                trades_to_close.append(
                                    (trade, current_price, current_date),
                                )

                # ポジションクローズ
                for trade, exit_price, exit_date in trades_to_close:
                    cash += trade.quantity * exit_price
                    if trade.symbol in positions:
                        positions[trade.symbol] -= trade.quantity
                        if positions[trade.symbol] <= 0:
                            del positions[trade.symbol]

                    trade.exit_date = exit_date
                    trade.exit_price = exit_price
                    trade.return_pct = (
                        exit_price - trade.entry_price
                    ) / trade.entry_price
                    trade.holding_days = (
                        exit_date.date() - trade.entry_date.date()
                    ).days

                # リバランス実行（指定頻度で）
                if (
                    len(trading_days) > 0
                    and (current_date - trading_days[0]).days % rebalance_frequency == 0
                ):
                    new_trades = self._rebalance_portfolio(
                        predictor,
                        symbols,
                        current_date,
                        all_data,
                        cash,
                        top_n,
                        score_threshold,
                    )

                    for trade in new_trades:
                        if trade.symbol in all_data:
                            entry_price = self._get_price_on_date(
                                all_data[trade.symbol], current_date,
                            )
                            if (
                                entry_price is not None
                                and cash >= trade.quantity * entry_price
                            ):
                                trade.entry_price = entry_price
                                cash -= trade.quantity * entry_price

                                if trade.symbol in positions:
                                    positions[trade.symbol] += trade.quantity
                                else:
                                    positions[trade.symbol] = trade.quantity

                                trades.append(trade)

            except Exception as e:
                logger.error(f"Error processing date {current_date}: {e!s}")
                continue

        # 最終的に残っているポジションをクローズ
        final_date = trading_days[-1] if len(trading_days) > 0 else end_dt
        for trade in trades:
            if trade.exit_date is None:
                if trade.symbol in all_data:
                    final_price = self._get_price_on_date(
                        all_data[trade.symbol], final_date,
                    )
                    if final_price is not None:
                        trade.exit_date = final_date
                        trade.exit_price = final_price
                        trade.return_pct = (
                            final_price - trade.entry_price
                        ) / trade.entry_price
                        trade.holding_days = (
                            final_date.date() - trade.entry_date.date()
                        ).days

        # バックテスト結果を計算
        return self._calculate_backtest_metrics(portfolio_values, trades)

    def _get_price_on_date(
        self, data: pd.DataFrame, date: pd.Timestamp,
    ) -> Optional[float]:
        """指定日の価格を取得"""
        try:
            # 完全一致を試行
            if date in data.index:
                return float(data.loc[date, "Close"])

            # 最も近い過去の日付を使用
            available_dates = data.index[data.index <= date]
            if len(available_dates) > 0:
                closest_date = available_dates.max()
                return float(data.loc[closest_date, "Close"])

            return None
        except Exception:
            return None

    def _rebalance_portfolio(
        self,
        predictor,
        symbols: List[str],
        current_date: pd.Timestamp,
        all_data: Dict[str, pd.DataFrame],
        available_cash: float,
        top_n: int,
        score_threshold: float = 60,  # 新しいパラメータ
    ) -> List[Trade]:
        """ポートフォリオをリバランス"""
        new_trades = []

        try:
            # 各銘柄のスコアを計算
            scores = {}
            for symbol in symbols:
                try:
                    if isinstance(predictor, MLStockPredictor):
                        # 機械学習モデルの場合、歴史的データを使用
                        historical_data = all_data[symbol][
                            all_data[symbol].index <= current_date
                        ]
                        if len(historical_data) >= 100:  # 十分なデータがある場合のみ
                            score = predictor.predict_score(symbol)
                        else:
                            score = 50.0
                    else:
                        # 従来のルールベースモデル
                        score = predictor.calculate_score(symbol)

                    scores[symbol] = score
                except Exception as e:
                    logger.warning(f"Error calculating score for {symbol}: {e!s}")
                    scores[symbol] = 50.0

            # スコア順にソート
            sorted_symbols = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_symbols = sorted_symbols[:top_n]

            # 上位銘柄に投資
            if available_cash > 0 and top_symbols:
                cash_per_stock = available_cash / len(top_symbols)

                for symbol, score in top_symbols:
                    if score > score_threshold:  # 調整可能な閾値
                        current_price = self._get_price_on_date(
                            all_data[symbol], current_date,
                        )
                        if current_price is not None and current_price > 0:
                            quantity = int(cash_per_stock / current_price)
                            if quantity > 0:
                                trade = Trade(
                                    symbol=symbol,
                                    entry_date=current_date,
                                    exit_date=None,
                                    entry_price=current_price,
                                    exit_price=None,
                                    quantity=quantity,
                                    trade_type="buy",
                                    score=score,
                                )
                                new_trades.append(trade)

        except Exception as e:
            logger.error(f"Error in rebalancing: {e!s}")

        return new_trades

    def _calculate_backtest_metrics(
        self, portfolio_values: List[Dict], trades: List[Trade],
    ) -> BacktestResult:
        """バックテスト指標を計算"""
        if not portfolio_values or not trades:
            return BacktestResult(
                total_return=0.0,
                annualized_return=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                win_rate=0.0,
                total_trades=0,
                avg_holding_days=0.0,
                best_trade=0.0,
                worst_trade=0.0,
                trades=[],
            )

        # ポートフォリオ価値の推移
        df_portfolio = pd.DataFrame(portfolio_values)
        df_portfolio.set_index("date", inplace=True)

        # リターン計算
        final_value = df_portfolio["portfolio_value"].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital

        # 年率リターン
        days = (df_portfolio.index[-1] - df_portfolio.index[0]).days
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # 最大ドローダウン
        cumulative_returns = df_portfolio["portfolio_value"] / self.initial_capital
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = abs(drawdowns.min())

        # 日次リターン
        daily_returns = df_portfolio["portfolio_value"].pct_change().dropna()

        # シャープレシオ
        if daily_returns.std() > 0:
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        # 取引統計
        completed_trades = [
            t for t in trades if t.exit_date is not None and t.return_pct is not None
        ]

        if completed_trades:
            returns = [t.return_pct for t in completed_trades]
            win_rate = len([r for r in returns if r > 0]) / len(returns)
            avg_holding_days = np.mean(
                [t.holding_days for t in completed_trades if t.holding_days is not None],
            )
            best_trade = max(returns)
            worst_trade = min(returns)
        else:
            win_rate = 0.0
            avg_holding_days = 0.0
            best_trade = 0.0
            worst_trade = 0.0

        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            total_trades=len(completed_trades),
            avg_holding_days=avg_holding_days,
            best_trade=best_trade,
            worst_trade=worst_trade,
            trades=completed_trades,
        )

    def compare_models(
        self,
        models: List[Tuple[str, object]],
        symbols: List[str],
        start_date: str,
        end_date: str,
    ) -> Dict[str, BacktestResult]:
        """複数モデルの比較"""
        results = {}

        for model_name, model in models:
            try:
                logger.info(f"Running backtest for {model_name}")
                result = self.run_backtest(
                    predictor=model,
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date,
                )
                results[model_name] = result

                logger.info(f"{model_name} - Total Return: {result.total_return:.2%}")
                logger.info(f"{model_name} - Sharpe Ratio: {result.sharpe_ratio:.2f}")

            except Exception as e:
                logger.error(f"Error running backtest for {model_name}: {e!s}")
                continue

        return results
