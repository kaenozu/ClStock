import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from data.stock_data import StockDataProvider
from config.settings import get_settings

# full_auto_system から必要なクラスをインポート
from full_auto_system import HybridPredictorAdapter, SentimentAnalyzerAdapter, RiskManagerAdapter, StrategyGeneratorAdapter
from full_auto_system import AutoRecommendation

class BacktestResult:
    """
    バックテスト結果を格納するクラス
    """
    def __init__(self):
        self.trades: List[Dict] = []  # 取引履歴
        self.portfolio_values: List[Tuple[datetime, float]] = []  # ポートフォリオの価値推移
        self.initial_capital: float = 0.0
        self.final_capital: float = 0.0
        self.total_return: float = 0.0
        self.annual_return: float = 0.0
        self.volatility: float = 0.0
        self.sharpe_ratio: float = 0.0
        self.max_drawdown: float = 0.0
        self.win_rate: float = 0.0

class BacktestEngine:
    """
    バックテストエンジンクラス
    """
    def __init__(self, initial_capital: Optional[float] = None):
        settings = get_settings()
        self.initial_capital = initial_capital or settings.backtest.default_initial_capital
        self.current_capital = self.initial_capital
        self.holdings: Dict[str, int] = {}  # 株の保有数 {'000001': 100, '000002': 200, ...}
        self.stock_data_cache: Dict[str, pd.DataFrame] = {}  # 各銘柄の株価データキャッシュ
        self.transaction_history: List[Dict] = []
        self.portfolio_history: List[Tuple[datetime, float]] = []
        self.data_provider = StockDataProvider()  # StockDataProvider インスタンス
        # full_auto_system のアダプターを初期化
        self.predictor = HybridPredictorAdapter()
        self.sentiment_analyzer = SentimentAnalyzerAdapter()
        self.risk_manager = RiskManagerAdapter()
        self.strategy_generator = StrategyGeneratorAdapter()
        
        # 設定からその他のパラメータを取得
        self.default_stop_loss_pct = settings.backtest.default_stop_loss_pct
        self.default_take_profit_pct = settings.backtest.default_take_profit_pct
        self.default_max_holding_days = settings.backtest.default_max_holding_days
        self.default_score_threshold = settings.backtest.default_score_threshold

    def load_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        指定された期間の株価データを読み込む
        yfinance は period でしか期間を指定できないため、start_date と end_date を period に変換して取得し、範囲を絞る
        """
        # yfinance は period でしか期間を指定できないため、start_date と end_date から period を推定
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        delta = end - start
        days = delta.days
        
        # 期間に応じて適切な period を選択
        if days <= 5:
            period = "5d"
        elif days <= 30:
            period = "1mo"
        elif days <= 90:
            period = "3mo"
        elif days <= 180:
            period = "6mo"
        elif days <= 365:
            period = "1y"
        elif days <= 730:
            period = "2y"
        else:
            period = "5y"
        
        # StockDataProvider を使用してデータ取得
        df = self.data_provider.get_stock_data(symbol, period=period)
        
        # 指定された範囲に絞る
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        
        if df.empty:
            raise ValueError(f"No data found for symbol {symbol} between {start_date} and {end_date}")
            
        print(f"Loaded stock data for {symbol} from {start_date} to {end_date}: {len(df)} records")
        return df

    def buy_stock(self, symbol: str, quantity: int, price: float, date: datetime, fee_rate: float = 0.0, tax_rate: float = 0.0):
        """
        株を購入する
        fee_rate: 手数料率 (例: 0.005 で 0.5%)
        tax_rate: 税率 (日本では株式売買に係る税金は基本0だが、消費税等で考慮する場合用)
        """
        # 手数料を含めた購入金額
        cost_before_fee = price * quantity
        fee = cost_before_fee * fee_rate
        cost = cost_before_fee + fee
        
        if cost <= self.current_capital:
            self.current_capital -= cost
            self.holdings[symbol] = self.holdings.get(symbol, 0) + quantity
            # 取引履歴に追加
            self.transaction_history.append({
                'date': date,
                'symbol': symbol,
                'action': 'BUY',
                'quantity': quantity,
                'price': price,
                'cost_before_fee': cost_before_fee,
                'fee': fee,
                'total_cost': cost
            })
            print(f"Bought {quantity} shares of {symbol} at {price} yen on {date}, fee: {fee} yen, total cost: {cost} yen")
        else:
            print(f"Not enough capital to buy {quantity} shares of {symbol} at {price} yen. Current capital: {self.current_capital}, required: {cost} yen")

    def sell_stock(self, symbol: str, quantity: int, price: float, date: datetime, fee_rate: float = 0.0, tax_rate: float = 0.0):
        """
        株を売却する
        fee_rate: 手数料率 (例: 0.005 で 0.5%)
        tax_rate: 税率 (売買益に対する税率、現在日本では0%だが将来の変更を考慮)
        """
        if symbol in self.holdings and self.holdings[symbol] >= quantity:
            revenue_before_fee_and_tax = price * quantity
            fee = revenue_before_fee_and_tax * fee_rate
            tax = 0  # 日本の株式売買益は現在税制優遇されているため0%と仮定
            if tax_rate > 0:
                # 必要に応じて税金の計算ロジックを追加
                # 例えば、売買益 = (売却価格 * 数量) - (購入価格 * 数量) - 手数料
                # tax = 売買益 * tax_rate
                pass
            
            revenue = revenue_before_fee_and_tax - fee - tax
            self.current_capital += revenue
            self.holdings[symbol] -= quantity
            if self.holdings[symbol] == 0:
                del self.holdings[symbol]
            # 取引履歴に追加
            self.transaction_history.append({
                'date': date,
                'symbol': symbol,
                'action': 'SELL',
                'quantity': quantity,
                'price': price,
                'revenue_before_fee_and_tax': revenue_before_fee_and_tax,
                'fee': fee,
                'tax': tax,
                'revenue': revenue
            })
            print(f"Sold {quantity} shares of {symbol} at {price} yen on {date}, fee: {fee} yen, tax: {tax} yen, net revenue: {revenue} yen")
        else:
            print(f"Not enough shares to sell {quantity} of {symbol}. Current holdings: {self.holdings.get(symbol, 0)}")

    def calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        現在の株価に基づいてポートフォリオの価値を計算する
        """
        total_value = self.current_capital
        for symbol, quantity in self.holdings.items():
            if symbol in current_prices:
                total_value += current_prices[symbol] * quantity
        return total_value

    def execute_strategy(self, symbols_to_trade: List[str], period_start: str, period_end: str, strategy_func = None):
        """
        full_auto_system の戦略に基づいてバックテストを実行する
        strategy_func: カスタム戦略関数 (未指定の場合は full_auto_system の戦略を使用)
        """
        print(f"Executing strategy for symbols: {symbols_to_trade} from {period_start} to {period_end}")
        # 日付範囲でループ
        current_date = datetime.strptime(period_start, '%Y-%m-%d')
        end_date = datetime.strptime(period_end, '%Y-%m-%d')

        # 各銘柄の直近のデータをキャッシュ
        latest_data_cache: Dict[str, pd.DataFrame] = {}

        while current_date <= end_date:
            # 土日をスキップ (0=月, 1=火, ..., 4=金, 5=土, 6=日)
            if current_date.weekday() < 5:
                # その日の価格データを取得
                current_prices = {}
                for symbol in symbols_to_trade:
                    try:
                        # 1日分のデータを取得
                        df = self.load_stock_data(symbol, current_date.strftime('%Y-%m-%d'), current_date.strftime('%Y-%m-%d'))
                        if not df.empty:
                            # Open価格を注文価格として使用
                            current_price = df['Open'].iloc[0]
                            current_prices[symbol] = current_price
                            
                            # 最新データをキャッシュ
                            latest_data_cache[symbol] = self.load_stock_data(symbol, (current_date - pd.Timedelta(days=365)).strftime('%Y-%m-%d'), current_date.strftime('%Y-%m-%d'))
                        else:
                            # データがなければ、前日の価格を取得
                            # キャッシュを確認
                            if symbol in self.stock_data_cache and not self.stock_data_cache[symbol].empty:
                                prev_day_data = self.stock_data_cache[symbol]
                                # current_dateより前のデータを取得
                                prev_data = prev_day_data[prev_day_data.index < current_date.strftime('%Y-%m-%d')]
                                if not prev_data.empty:
                                    prev_price = prev_data['Close'].iloc[-1]
                                    current_prices[symbol] = prev_price
                                    print(f"Using previous price {prev_price} for {symbol} on {current_date.strftime('%Y-%m-%d')}")
                                else:
                                    print(f"No previous data available for {symbol} before {current_date.strftime('%Y-%m-%d')}")
                            else:
                                print(f"No data available for {symbol} on {current_date.strftime('%Y-%m-%d')}")
                    except Exception as e:
                        print(f"Error loading data for {symbol} on {current_date.strftime('%Y-%m-%d')}: {e}")
                
                if strategy_func:
                    # カスタム戦略関数を呼び出す
                    recommended_actions = strategy_func(current_date, current_prices)
                    for symbol, action in recommended_actions.items():
                        if symbol in current_prices:
                            current_price = current_prices[symbol]
                            # 仮の数量 (実際には戦略関数が数量も返すべき)
                            quantity = 10
                            if action == "BUY":
                                self.buy_stock(symbol, quantity, current_price, current_date, fee_rate=0.005)
                            elif action == "SELL":
                                self.sell_stock(symbol, quantity, current_price, current_date, fee_rate=0.005)
                else:
                    # full_auto_system の戦略を使用
                    # 現在の価格に基づいて、推奨銘柄を取得
                    for symbol in symbols_to_trade:
                        if symbol in latest_data_cache and symbol in current_prices:
                            data = latest_data_cache[symbol]
                            current_price = current_prices[symbol]
                            
                            # full_auto_system の内部ロジックを模倣して戦略を生成
                            # _analyze_single_stock の一部を実行
                            try:
                                # 1. 予測モデル適用
                                predictions = self.predictor.predict(symbol, data)
                                if not predictions:
                                    continue

                                # 2. リスク分析
                                risk_analysis = self.risk_manager.analyze_risk(symbol, data, predictions)
                                # 3. 感情分析
                                sentiment_result = self.sentiment_analyzer.analyze_sentiment(symbol)
                                # 4. 戦略生成
                                strategy = self.strategy_generator.generate_strategy(
                                    symbol, data, predictions, risk_analysis, sentiment_result
                                )

                                if strategy:
                                    # 戦略に基づいてアクションを決定
                                    # 例: target_price が current_price より高い場合は BUY、低い場合は SELL
                                    entry_price = strategy.get('entry_price', current_price)
                                    target_price = strategy.get('target_price', current_price)
                                    # 単純化: target_price が entry_price より高い場合は買い、低い場合は売り
                                    # また、現在価格がエントリ価格より低い場合に買いとかも考えられる
                                    if target_price > entry_price and current_price <= entry_price:
                                        action = 'BUY'
                                        # 仮の数量 (実際には戦略関数が数量も返すべき)
                                        quantity = 10
                                        self.buy_stock(symbol, quantity, current_price, current_date, fee_rate=0.005)
                                    elif target_price < entry_price and current_price >= entry_price:
                                        action = 'SELL'
                                        # 仮の数量 (実際には戦略関数が数量も返すべき)
                                        quantity = 10
                                        self.sell_stock(symbol, quantity, current_price, current_date, fee_rate=0.005)
                                else:
                                    print(f"No strategy generated for {symbol} on {current_date.strftime('%Y-%m-%d')}, skipping...")
                            except Exception as e:
                                print(f"Error generating strategy for {symbol} on {current_date.strftime('%Y-%m-%d')}: {e}")
                
                # ポートフォリオ履歴を記録
                portfolio_value = self.calculate_portfolio_value(current_prices)
                self.portfolio_history.append((current_date, portfolio_value))

            current_date += pd.Timedelta(days=1)

    def calculate_metrics(self) -> BacktestResult:
        """
        バックテスト結果の評価指標を計算する
        """
        result = BacktestResult()
        result.initial_capital = self.initial_capital
        if self.portfolio_history:
            result.final_capital = self.portfolio_history[-1][1]
            result.total_return = (result.final_capital - result.initial_capital) / result.initial_capital

            # 年率リターンの計算
            if len(self.portfolio_history) > 0:
                start_date = self.portfolio_history[0][0]
                end_date = self.portfolio_history[-1][0]
                years = (end_date - start_date).days / 365.25
                if years > 0:
                    result.annual_return = (result.final_capital / result.initial_capital) ** (1 / years) - 1

            # リターンの計算 (日々のリターン)
            if len(self.portfolio_history) > 1:
                values = [v for _, v in self.portfolio_history]
                daily_returns = [0.0] + [(v2 - v1) / v1 for v1, v2 in zip(values[:-1], values[1:])]
                
                # 年間リターン (算術平均)
                result.annual_return = np.mean(daily_returns) * 252  # 252は年間取引日数
                
                # ボラティリティ（リターンの標準偏差）
                result.volatility = np.std(daily_returns) * np.sqrt(252)
                
                # シャープレシオ (無リスクレートを例として2%とする)
                risk_free_rate = 0.02
                result.sharpe_ratio = (result.annual_return - risk_free_rate) / result.volatility if result.volatility != 0 else 0.0

                # 最大ドローダウンの計算
                running_max = np.maximum.accumulate(values)
                drawdown = (values - running_max) / running_max
                result.max_drawdown = np.min(drawdown) if drawdown.size > 0 else 0.0
                
                # 勝率の計算 (取引履歴から)
                # 買いポジションの損益を計算 (売却時)
                buy_transactions = [t for t in self.transaction_history if t['action'] == 'BUY']
                sell_transactions = [t for t in self.transaction_history if t['action'] == 'SELL']
                
                # 各売却に対して、対応する買いを特定し、損益を計算 (簡易版)
                # 実際には、FIFO (先入先出) で計算すべき
                profit_count = 0
                total_trades = 0
                for sell in sell_transactions:
                    # 売却数量に対応する買いを特定 (簡易版: 最も古い買いから)
                    remaining_quantity = sell['quantity']
                    for buy in buy_transactions:
                        if buy['symbol'] == sell['symbol'] and buy['quantity'] > 0:
                            matched_quantity = min(remaining_quantity, buy['quantity'])
                            buy['quantity'] -= matched_quantity
                            remaining_quantity -= matched_quantity
                            if buy['quantity'] <= 0:
                                buy['quantity'] = 0  # 表示用に0に設定 (実際の資産とは関係ない)
                            
                            # 損益計算 (売却価格 - 購入価格) * 数量 - 手数料
                            # 簡略化: 手数料を無視し、均等価格で計算
                            avg_buy_price = buy['price'] if 'price' in buy else buy.get('total_cost', buy.get('cost_before_fee', 0)) / buy.get('quantity', 1)
                            profit = (sell['price'] - avg_buy_price) * matched_quantity
                            if profit > 0:
                                profit_count += 1
                            total_trades += 1
                            
                            if remaining_quantity == 0:
                                break
                    if remaining_quantity > 0:
                        print(f"Warning: Could not match all quantities for sell transaction of {sell['quantity']} shares of {sell['symbol']}")
                
                result.win_rate = profit_count / total_trades if total_trades > 0 else 0.0

        result.trades = self.transaction_history
        result.portfolio_values = self.portfolio_history

        return result

    def plot_results(self, result: BacktestResult):
        """
        バックテスト結果をプロットする
        """
        if not result.portfolio_values:
            print("No portfolio values to plot.")
            return

        dates = [date for date, _ in result.portfolio_values]
        values = [value for _, value in result.portfolio_values]

        # グラフの設定
        fig, ax = plt.subplots(figsize=(12, 8))

        # ポートフォリオ価値の推移
        ax.plot(dates, values, label='Portfolio Value', color='blue')

        # 最大ドローダウンの範囲を塗りつぶす
        running_max = np.maximum.accumulate(values)
        drawdown = (values - running_max) / running_max
        ax.fill_between(dates, values, running_max, where=(drawdown < 0), color='red', alpha=0.3, label='Drawdown')

        # グラフの装飾
        ax.set_title('Backtest Portfolio Value Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value (JPY)')
        ax.legend()
        ax.grid(True)

        # X軸のフォーマットを日付に
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

    def generate_summary_report(self, result: BacktestResult):
        """
        バックテスト結果の要約レポートを生成する
        """
        print("="*50)
        print("BACKTEST SUMMARY REPORT")
        print("="*50)
        print(f"Initial Capital: JPY {result.initial_capital:,.2f}")
        print(f"Final Capital: JPY {result.final_capital:,.2f}")
        print(f"Total Return: {result.total_return:.2%}")
        print(f"Annual Return: {result.annual_return:.2%}")
        print(f"Volatility (Ann.): {result.volatility:.4f}")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.4f}")
        print(f"Max Drawdown: {result.max_drawdown:.2%}")
        print(f"Win Rate: {result.win_rate:.2%}")
        print(f"Total Trades: {len(result.trades)}")
        print("="*50)

    def run_backtest(self, strategy_func, period_start: str, period_end: str) -> BacktestResult:
        """
        バックテストを実行して結果を返す
        """
        print(f"Starting backtest from {period_start} to {period_end}")
        # 戦略を実行
        self.execute_strategy(strategy_func, None, period_start, period_end)
        # 評価指標を計算
        result = self.calculate_metrics()
        print(f"Backtest completed. Final capital: {result.final_capital}, Total return: {result.total_return:.2%}")
        return result

# 使用例
if __name__ == "__main__":
    # バックテストエンジンの初期化
    engine = BacktestEngine(initial_capital=1000000)

    # 戦略関数（ダミー）
    def dummy_strategy(date, current_prices):
        return {"000001": "BUY"}  # ダミーの推奨

    # バックテストの実行
    result = engine.run_backtest(dummy_strategy, "2023-01-01", "2023-12-31")

    # 結果の表示
    print(f"Total Return: {result.total_return:.2%}")
    print(f"Annual Return: {result.annual_return:.2%}")
    print(f"Volatility: {result.volatility:.4f}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.4f}")
    print(f"Max Drawdown: {result.max_drawdown:.2%}")