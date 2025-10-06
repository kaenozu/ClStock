import unittest
from datetime import datetime

from backtesting import BacktestEngine, BacktestResult


class TestBacktestEngine(unittest.TestCase):
    def setUp(self):
        # テスト用の一時的な初期資本
        self.test_initial_capital = 1000000.0
        # テスト用のエンジンを作成
        self.engine = BacktestEngine(initial_capital=self.test_initial_capital)

    def test_initialization(self):
        """初期化のテスト"""
        self.assertEqual(self.engine.initial_capital, self.test_initial_capital)
        self.assertEqual(self.engine.current_capital, self.test_initial_capital)
        self.assertEqual(self.engine.holdings, {})
        self.assertEqual(self.engine.transaction_history, [])
        self.assertEqual(self.engine.portfolio_history, [])

    def test_buy_stock(self):
        """株式購入のテスト"""
        symbol = "000001"
        quantity = 10
        price = 1000.0
        date = datetime.now()

        # 購入実行
        self.engine.buy_stock(symbol, quantity, price, date, fee_rate=0.0)

        # 期待される結果
        expected_capital = self.test_initial_capital - (price * quantity)
        expected_holdings = {symbol: quantity}

        self.assertEqual(self.engine.current_capital, expected_capital)
        self.assertEqual(self.engine.holdings, expected_holdings)
        self.assertEqual(len(self.engine.transaction_history), 1)
        self.assertEqual(self.engine.transaction_history[0]["action"], "BUY")

    def test_sell_stock(self):
        """株式売却のテスト"""
        symbol = "000001"
        buy_quantity = 20
        sell_quantity = 10
        buy_price = 1000.0
        sell_price = 1100.0
        date = datetime.now()

        # まず購入
        self.engine.buy_stock(symbol, buy_quantity, buy_price, date, fee_rate=0.0)

        # 売却実行
        self.engine.sell_stock(symbol, sell_quantity, sell_price, date, fee_rate=0.0)

        # 期待される結果
        remaining_quantity = buy_quantity - sell_quantity
        expected_capital = (
            self.test_initial_capital
            - (buy_price * buy_quantity)
            + (sell_price * sell_quantity)
        )
        expected_holdings = {symbol: remaining_quantity}

        self.assertEqual(self.engine.current_capital, expected_capital)
        self.assertEqual(self.engine.holdings[symbol], expected_holdings[symbol])
        self.assertEqual(len(self.engine.transaction_history), 2)
        self.assertEqual(self.engine.transaction_history[1]["action"], "SELL")

    def test_calculate_portfolio_value(self):
        """ポートフォリオ価値計算のテスト"""
        # 購入
        self.engine.buy_stock("000001", 10, 1000.0, datetime.now(), fee_rate=0.0)
        self.engine.buy_stock("000002", 5, 2000.0, datetime.now(), fee_rate=0.0)

        # 現在価格
        current_prices = {"000001": 1100.0, "000002": 1900.0}

        # 計算
        portfolio_value = self.engine.calculate_portfolio_value(current_prices)

        # 期待される価値: 残りの現金 + 株式の価値
        expected_cash = self.test_initial_capital - (10 * 1000.0) - (5 * 2000.0)
        expected_stock_value = (10 * 1100.0) + (5 * 1900.0)
        expected_portfolio_value = expected_cash + expected_stock_value

        self.assertEqual(portfolio_value, expected_portfolio_value)

    def test_calculate_metrics(self):
        """評価指標計算のテスト"""
        # いくつかの取引履歴を追加
        self.engine.transaction_history = [
            {
                "symbol": "000001",
                "action": "BUY",
                "quantity": 10,
                "price": 1000,
                "total_cost": 10000,
            },
            {
                "symbol": "000001",
                "action": "SELL",
                "quantity": 5,
                "price": 1100,
                "revenue": 5500,
            },
            {
                "symbol": "000001",
                "action": "SELL",
                "quantity": 5,
                "price": 900,
                "revenue": 4500,
            },
        ]

        # ポートフォリオ履歴を追加
        self.engine.portfolio_history = [
            (datetime(2023, 1, 1), self.test_initial_capital),
            (datetime(2023, 1, 2), self.test_initial_capital * 1.01),
            (datetime(2023, 1, 3), self.test_initial_capital * 0.99),  # ドローダウン
            (datetime(2023, 1, 4), self.test_initial_capital * 1.02),
        ]

        # 計算
        result = self.engine.calculate_metrics()

        # 結果の検証
        self.assertIsInstance(result, BacktestResult)
        self.assertEqual(result.initial_capital, self.test_initial_capital)
        self.assertEqual(result.win_rate, 0.5)  # 2取引中1勝利 = 50%
        # 他の指標も必要に応じて検証


if __name__ == "__main__":
    unittest.main()
