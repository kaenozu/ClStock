#!/usr/bin/env python3

import argparse
import asyncio
import sys
from datetime import datetime

from data.stock_data import StockDataProvider
from models.predictor import StockPredictor
from models.recommendation import StockRecommendation

# Windows環境でのUnicode出力設定
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def format_currency(amount: float) -> str:
    return f"{amount:,.0f}円"


def format_percentage(amount: float, base: float) -> str:
    percentage = ((amount - base) / base) * 100
    return f"{percentage:+.1f}%"


def print_recommendation(rec: StockRecommendation):
    print(f"[{rec.rank}位] {rec.company_name} ({rec.symbol})")
    print(f"   [買] 買うタイミング: {rec.buy_timing}")
    print(f"   [価] 現在価格: {format_currency(rec.current_price)}")
    print(f"   [目] 目標価格: {format_currency(rec.target_price)}")
    stop_loss_pct = format_percentage(rec.stop_loss, rec.current_price)
    print(f"   [損] 損切り目安: {format_currency(rec.stop_loss)} ({stop_loss_pct})")
    profit_1_pct = format_percentage(rec.profit_target_1, rec.current_price)
    profit_2_pct = format_percentage(rec.profit_target_2, rec.current_price)
    profit_1_str = f"{format_currency(rec.profit_target_1)} ({profit_1_pct})"
    profit_2_str = f"{format_currency(rec.profit_target_2)} ({profit_2_pct})"
    print(f"   [利] 利益目標: {profit_1_str}、{profit_2_str}")
    print(f"   [期] 保有期間の目安: {rec.holding_period}")
    print(f"   [点] 推奨度スコア: {rec.score:.1f}/100")
    print(f"   [理] 推奨理由: {rec.recommendation_reason}")
    print()


def print_header():
    print("=" * 60)
    print("★ 今週のおすすめ銘柄（30〜90日向け）")
    print("=" * 60)
    print()


def print_footer():
    print("=" * 60)
    print(f"[時] 生成日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M')}")
    warning_msg = "この情報は投資判断の参考であり、投資は自己責任で行ってください。"
    print(f"[注] 注意: {warning_msg}")
    print("=" * 60)


async def main():
    parser = argparse.ArgumentParser(
        description="ClStock - 中期的な推奨銘柄予想システム",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="表示する推奨銘柄の上位N件 (デフォルト: 5)",
    )
    parser.add_argument("--symbol", type=str, help="特定の銘柄コードの推奨情報を表示")
    parser.add_argument("--list", action="store_true", help="利用可能な銘柄一覧を表示")
    parser.add_argument("--json", action="store_true", help="JSON形式で出力")

    args = parser.parse_args()

    try:
        if args.list:
            data_provider = StockDataProvider()
            print("[一覧] 利用可能な銘柄一覧:")
            print("-" * 40)
            for symbol, name in data_provider.jp_stock_codes.items():
                print(f"{symbol}: {name}")
            return

        predictor = StockPredictor()

        if args.symbol:
            try:
                recommendation = predictor.generate_recommendation(args.symbol)
                recommendation.rank = 1

                if args.json:
                    import json

                    json_data = recommendation.dict()
                    print(json.dumps(json_data, ensure_ascii=False, indent=2))
                else:
                    print_header()
                    print_recommendation(recommendation)
                    print_footer()
            except Exception as e:
                error_msg = (
                    f"銘柄 {args.symbol} の推奨情報を取得できませんでした: {e!s}"
                )
                print(f"[ERROR] エラー: {error_msg}")
                return

        else:
            recommendations = predictor.get_top_recommendations(args.top)

            if args.json:
                import json

                data = {
                    "recommendations": [rec.dict() for rec in recommendations],
                    "generated_at": datetime.now().isoformat(),
                    "market_status": (
                        "市場営業時間外"
                        if datetime.now().hour < 9 or datetime.now().hour > 15
                        else "市場営業中"
                    ),
                }
                print(json.dumps(data, ensure_ascii=False, indent=2))
            else:
                print_header()
                for rec in recommendations:
                    print_recommendation(rec)
                print_footer()

    except KeyboardInterrupt:
        print("\n[WARN] 処理が中断されました。")
    except Exception as e:
        print(f"[ERROR] エラーが発生しました: {e!s}")


if __name__ == "__main__":
    asyncio.run(main())
