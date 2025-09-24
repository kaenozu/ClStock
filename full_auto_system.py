#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
繝輔Ν繧ｪ繝ｼ繝域兜雉・す繧ｹ繝・Β
螳悟・閾ｪ蜍募喧・啜SE4000譛驕ｩ蛹・竊・蟄ｦ鄙偵・險鍋ｷｴ 竊・螢ｲ雋ｷ繧ｿ繧､繝溘Φ繧ｰ謠千､ｺ
"""

import logging
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import asyncio
import sys

# 譌｢蟄倥す繧ｹ繝・Β縺ｮ繧､繝ｳ繝昴・繝・
from tse_4000_optimizer import TSE4000Optimizer
from models_new.hybrid.hybrid_predictor import HybridStockPredictor
from models_new.hybrid.prediction_modes import PredictionMode
from models_new.advanced.market_sentiment_analyzer import MarketSentimentAnalyzer
from models_new.advanced.trading_strategy_generator import (
    AutoTradingStrategyGenerator,
    ActionType,
)
from models_new.advanced.risk_management_framework import RiskManager
from data.stock_data import StockDataProvider


@dataclass
class AutoRecommendation:
    """閾ｪ蜍墓兜雉・耳螂ｨ"""

    symbol: str
    company_name: str
    action: ActionType
    entry_price: float
    target_price: float
    stop_loss: float
    buy_date: datetime
    sell_date: datetime
    expected_return: float
    confidence: float
    reasoning: str
    risk_level: str
    recommendation_score: float
    risk_score: float
    buy_timing: datetime = field(init=False)
    sell_timing: datetime = field(init=False)

    def __post_init__(self):
        self.buy_timing = self.buy_date
        self.sell_timing = self.sell_date


class FullAutoInvestmentSystem:
    """
    繝輔Ν繧ｪ繝ｼ繝域兜雉・す繧ｹ繝・Β

    迚ｹ蠕ｴ:
    - 螳悟・閾ｪ蜍募喧・医Θ繝ｼ繧ｶ繝ｼ蛻､譁ｭ荳崎ｦ・ｼ・
    - TSE4000譛驕ｩ蛹冶・蜍募ｮ溯｡・
    - 蟄ｦ鄙偵・險鍋ｷｴ閾ｪ蜍募ｮ滓命
    - 螢ｲ雋ｷ繧ｿ繧､繝溘Φ繧ｰ閾ｪ蜍慕ｮ怜・
    """

    _model_preparation_attempted = False
    _model_preparation_success = False

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # 繧ｵ繝悶す繧ｹ繝・Β蛻晄悄蛹・
        self.tse_optimizer = TSE4000Optimizer()
        self.hybrid_predictor = HybridStockPredictor(
            enable_cache=True,
            enable_adaptive_optimization=True,
            enable_streaming=True,
            enable_multi_gpu=True,
            enable_real_time_learning=True,
        )
        self.sentiment_analyzer = MarketSentimentAnalyzer()
        self.strategy_generator = AutoTradingStrategyGenerator()
        self.risk_manager = RiskManager()
        self.data_provider = StockDataProvider()


        # 閾ｪ蜍募喧險ｭ螳・
        self.auto_settings = {
            "portfolio_size": 10,  # 謗ｨ螂ｨ驫俶氛謨ｰ
            "investment_period_days": 30,  # 謚戊ｳ・悄髢難ｼ域律・・
            "min_confidence": 0.7,  # 譛蟆丈ｿ｡鬆ｼ蠎ｦ
            "max_risk_score": 2.5,  # 譛螟ｧ繝ｪ繧ｹ繧ｯ繧ｹ繧ｳ繧｢
            "rebalance_threshold": 0.1,  # 繝ｪ繝舌Λ繝ｳ繧ｹ髢ｾ蛟､
            "model_refresh_days": 30,  # 繝｢繝・Ν蟷ｴ逧・繧､繝ｳ繝悶Ν繧ｯ逕ｻ驛ｨ
        }

        if not FullAutoInvestmentSystem._model_preparation_attempted:
            FullAutoInvestmentSystem._model_preparation_success = self._ensure_models_ready()
            FullAutoInvestmentSystem._model_preparation_attempted = True
        elif not FullAutoInvestmentSystem._model_preparation_success:
            FullAutoInvestmentSystem._model_preparation_success = self._ensure_models_ready()

        self.logger.info("FullAutoInvestmentSystem initialized")

    async def run_full_auto_analysis(self) -> List[AutoRecommendation]:
        """繝輔Ν繧ｪ繝ｼ繝亥・譫仙ｮ溯｡・""
        self.logger.info("噫 繝輔Ν繧ｪ繝ｼ繝域兜雉・す繧ｹ繝・Β髢句ｧ・)

        # 蟶ょｴ譎る俣繝√ぉ繝・け
        if not self._show_market_hours_warning():
            print("蜃ｦ逅・ｒ荳ｭ譁ｭ縺励∪縺励◆縲・)
            return []

        # 騾ｲ謐苓｡ｨ遉ｺ縺ｮ蛻晄悄蛹・
        total_steps = 4
        current_step = 0

        def show_progress(step_name: str, step_num: int):
            nonlocal current_step
            current_step = step_num
            progress = (current_step / total_steps) * 100
            print(
                f"\n[騾ｲ謐余 [{current_step}/{total_steps}] ({progress:.0f}%) - {step_name}"
            )
            print("=" * 60)

        try:
            # Step 1: TSE4000譛驕ｩ蛹厄ｼ亥ｿ・ｦ√↓蠢懊§縺ｦ・・
            show_progress("TSE4000譛驕ｩ蛹門ｮ溯｡御ｸｭ...", 1)
            optimized_symbols = await self._auto_tse4000_optimization()
            print(f"[螳御ｺ・ 譛驕ｩ蛹門ｮ御ｺ・ {len(optimized_symbols)}驫俶氛驕ｸ蜃ｺ")

            # Step 2: 蟄ｦ鄙偵・險鍋ｷｴ閾ｪ蜍募ｮ滓命
            show_progress("蟄ｦ鄙偵・險鍋ｷｴ螳溯｡御ｸｭ...", 2)
            await self._auto_learning_and_training(optimized_symbols)
            print("[螳御ｺ・ 蟄ｦ鄙偵・險鍋ｷｴ螳御ｺ・)

            # Step 3: 邱丞粋蛻・梵縺ｨ謗ｨ螂ｨ逕滓・
            show_progress("邱丞粋蛻・梵繝ｻ謗ｨ螂ｨ逕滓・荳ｭ...", 3)
            recommendations = await self._generate_auto_recommendations(
                optimized_symbols
            )
            print(f"[螳御ｺ・ 謗ｨ螂ｨ逕滓・螳御ｺ・ {len(recommendations)}莉ｶ")

            # Step 4: 邨先棡陦ｨ遉ｺ
            show_progress("邨先棡陦ｨ遉ｺ", 4)
            self._display_recommendations(recommendations)
            print("[螳御ｺ・ 繝輔Ν繧ｪ繝ｼ繝亥・譫仙ｮ御ｺ・ｼ・)

            return recommendations

        except Exception as e:
            print(f"\n[繧ｨ繝ｩ繝ｼ] 繝輔Ν繧ｪ繝ｼ繝亥・譫仙､ｱ謨・ {str(e)}")
            self.logger.error(f"繝輔Ν繧ｪ繝ｼ繝亥・譫仙､ｱ謨・ {str(e)}")
            return []

    async def _auto_tse4000_optimization(self) -> List[str]:
        """TSE4000 auto optimization with enhanced error handling"""
        self.logger.info("投 TSE4000譛驕ｩ蛹門ｮ溯｡御ｸｭ...")
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                # 蜑榊屓譛驕ｩ蛹悶°繧峨・邨碁℃譎る俣繝√ぉ繝・け
                need_optimization = self._check_optimization_necessity()

                if need_optimization:
                    print(
                        f"[螳溯｡珪 譛驕ｩ蛹悶′蠢・ｦ√〒縺吶ょｮ溯｡御ｸｭ... (隧ｦ陦・{retry_count + 1}/{max_retries})"
                    )

                    # TSE4000譛驕ｩ蛹門ｮ溯｡鯉ｼ医ち繧､繝繧｢繧ｦ繝井ｻ倥″・・
                    try:
                        optimization_result = await asyncio.wait_for(
                            asyncio.create_task(self._run_tse_optimization_async()),
                            timeout=300,  # 5蛻・ち繧､繝繧｢繧ｦ繝・
                        )
                    except asyncio.TimeoutError:
                        print("[繧ｿ繧､繝繧｢繧ｦ繝・ TSE4000譛驕ｩ蛹悶′繧ｿ繧､繝繧｢繧ｦ繝医＠縺ｾ縺励◆")
                        raise Exception("TSE4000譛驕ｩ蛹悶ち繧､繝繧｢繧ｦ繝・)

                    if (
                        optimization_result
                        and "optimized_portfolio" in optimization_result
                    ):
                        symbols = [
                            stock.symbol
                            for stock in optimization_result["optimized_portfolio"]
                        ]
                        selected_symbols = symbols[
                            : self.auto_settings["portfolio_size"]
                        ]

                        if len(selected_symbols) >= 5:  # 譛菴・驫俶氛縺ｯ蠢・ｦ・
                            # 譛驕ｩ蛹門ｱ･豁ｴ繧定ｨ倬鹸
                            self._save_optimization_history(selected_symbols)

                            print(f"[螳御ｺ・ 譛驕ｩ蛹門ｮ御ｺ・ {len(selected_symbols)}驫俶氛驕ｸ蜃ｺ")
                            self.logger.info(
                                f"笨・譛驕ｩ蛹門ｮ御ｺ・ {len(selected_symbols)}驫俶氛驕ｸ蜃ｺ"
                            )
                            return selected_symbols
                        else:
                            raise Exception(
                                f"驕ｸ蜃ｺ驫俶氛謨ｰ荳崎ｶｳ: {len(selected_symbols)}驫俶氛"
                            )
                    else:
                        raise Exception("譛驕ｩ蛹也ｵ先棡縺檎┌蜉ｹ縺ｾ縺溘・遨ｺ縺ｧ縺・)

                else:
                    print("[菴ｿ逕ｨ] 蜑榊屓縺ｮ譛驕ｩ蛹也ｵ先棡繧剃ｽｿ逕ｨ")
                    self.logger.info("搭 蜑榊屓縺ｮ譛驕ｩ蛹也ｵ先棡繧剃ｽｿ逕ｨ")
                    # 蜑榊屓縺ｮ邨先棡繧定ｪｭ縺ｿ霎ｼ縺ｿ
                    previous_symbols = self._load_previous_optimization()
                    if len(previous_symbols) >= 5:
                        return previous_symbols
                    else:
                        print("[隴ｦ蜻馨 蜑榊屓邨先棡繧ゆｸ崎ｶｳ縲ゅョ繝輔か繝ｫ繝磯釜譟・ｽｿ逕ｨ")
                        return self._get_default_symbols()

            except Exception as e:
                retry_count += 1
                error_msg = str(e)
                print(
                    f"[繧ｨ繝ｩ繝ｼ] 譛驕ｩ蛹悶お繝ｩ繝ｼ (隧ｦ陦・{retry_count}/{max_retries}): {error_msg}"
                )
                self.logger.warning(
                    f"TSE4000譛驕ｩ蛹悶お繝ｩ繝ｼ (隧ｦ陦・{retry_count}): {error_msg}"
                )

                if retry_count < max_retries:
                    print(f"[蠕・ｩ歉 {5}遘貞ｾ後↓蜀崎ｩｦ陦後＠縺ｾ縺・..")
                    await asyncio.sleep(5)
                else:
                    print("[螳牙・] 譛螟ｧ隧ｦ陦悟屓謨ｰ蛻ｰ驕斐ゅョ繝輔か繝ｫ繝磯釜譟・ｒ菴ｿ逕ｨ縺励∪縺・)
                    self.logger.error(f"TSE4000譛驕ｩ蛹匁怙邨ょ､ｱ謨・ {error_msg}")
                    return self._get_default_symbols()

        return self._get_default_symbols()

    async def _run_tse_optimization_async(self):
        """TSE4000譛驕ｩ蛹悶・髱槫酔譛溷ｮ溯｡・""
        # 蜷梧悄逧・↑TSE4000譛驕ｩ蛹悶ｒ髱槫酔譛溘〒螳溯｡・
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.tse_optimizer.run_comprehensive_optimization
        )

    def _check_market_hours(self) -> Tuple[bool, str]:
        """蟶ょｴ譎る俣繝√ぉ繝・け"""
        try:
            current_time = datetime.now()
            weekday = current_time.weekday()  # 0=譛域屆, 6=譌･譖・
            hour = current_time.hour
            minute = current_time.minute

            # 蝨滓律縺ｯ蟶ょｴ莨第･ｭ
            if weekday >= 5:  # 蝨滓屆譌･繝ｻ譌･譖懈律
                next_monday = current_time + timedelta(days=(7 - weekday))
                return (
                    False,
                    f"蟶ょｴ莨第･ｭ譌･縺ｧ縺吶よｬ｡蝗樣幕蝣ｴ: {next_monday.strftime('%m/%d・域怦・・:00')}",
                )

            # 蟷ｳ譌･縺ｮ蜿門ｼ墓凾髢薙メ繧ｧ繝・け
            # 譚ｱ險ｼ: 9:00-11:30・亥燕蝣ｴ・峨・2:30-15:00・亥ｾ悟ｴ・・
            current_minutes = hour * 60 + minute

            # 蜑榊ｴ: 9:00-11:30 (540-690蛻・
            morning_start = 9 * 60  # 540蛻・
            morning_end = 11 * 60 + 30  # 690蛻・

            # 蠕悟ｴ: 12:30-15:00 (750-900蛻・
            afternoon_start = 12 * 60 + 30  # 750蛻・
            afternoon_end = 15 * 60  # 900蛻・

            if morning_start <= current_minutes <= morning_end:
                return True, "蜑榊ｴ蜿門ｼ墓凾髢謎ｸｭ"
            elif afternoon_start <= current_minutes <= afternoon_end:
                return True, "蠕悟ｴ蜿門ｼ墓凾髢謎ｸｭ"
            elif current_minutes < morning_start:
                return False, f"蟶ょｴ髢句ｴ蜑阪〒縺吶る幕蝣ｴ譎ょ綾: 9:00"
            elif morning_end < current_minutes < afternoon_start:
                return False, f"譏ｼ莨代∩譎る俣縺ｧ縺吶ょｾ悟ｴ髢句ｧ・ 12:30"
            else:  # current_minutes > afternoon_end
                next_day = current_time + timedelta(days=1)
                if next_day.weekday() >= 5:  # 鄙梧律縺悟悄譌･
                    next_monday = current_time + timedelta(
                        days=(7 - current_time.weekday())
                    )
                    return (
                        False,
                        f"蟶ょｴ邨ゆｺ・ｾ後〒縺吶よｬ｡蝗樣幕蝣ｴ: {next_monday.strftime('%m/%d・域怦・・:00')}",
                    )
                else:
                    return (
                        False,
                        f"蟶ょｴ邨ゆｺ・ｾ後〒縺吶よｬ｡蝗樣幕蝣ｴ: {next_day.strftime('%m/%d 9:00')}",
                    )

        except Exception as e:
            self.logger.warning(f"蟶ょｴ譎る俣繝√ぉ繝・け繧ｨ繝ｩ繝ｼ: {e}")
            return True, "蟶ょｴ譎る俣繝√ぉ繝・け辟｡蜉ｹ・亥・逅・ｶ咏ｶ夲ｼ・

    def _show_market_hours_warning(self) -> bool:
        """蟶ょｴ譎る俣螟悶・隴ｦ蜻願｡ｨ遉ｺ"""
        is_open, message = self._check_market_hours()

        if not is_open:
            print(f"\n[隴ｦ蜻馨 {message}")
            print("[豕ｨ諢従 蟶ょｴ譎る俣螟悶〒繧ょ・譫舌・螳溯｡後〒縺阪∪縺吶′縲・)
            print("       螳滄圀縺ｮ蜿門ｼ輔・蟶ょｴ髢句ｴ譎る俣蜀・↓陦後▲縺ｦ縺上□縺輔＞縲・)

            while True:
                choice = input("\n邯夊｡後＠縺ｾ縺吶°・・(y/n): ").strip().lower()
                if choice in ["y", "yes", "縺ｯ縺・]:
                    return True
                elif choice in ["n", "no", "縺・＞縺・]:
                    return False
                else:
                    print("y縺ｾ縺溘・n縺ｧ蜈･蜉帙＠縺ｦ縺上□縺輔＞縲・)
        else:
            print(f"[OK] {message}")
            return True

    def _check_optimization_necessity(self) -> bool:
        """譛驕ｩ蛹門ｿ・ｦ∵ｧ蛻､螳・""
        try:
            # 譛驕ｩ蛹門ｱ･豁ｴ繝輔ぃ繧､繝ｫ繧堤｢ｺ隱・
            history_file = "tse4000_optimization_history.json"

            if not os.path.exists(history_file):
                # 螻･豁ｴ繝輔ぃ繧､繝ｫ縺後↑縺・ｴ蜷医・譛驕ｩ蛹門ｮ溯｡・
                return True

            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)

            if not history or "last_optimization" not in history:
                return True

            # 蜑榊屓譛驕ｩ蛹匁律譎ゅｒ蜿門ｾ・
            last_optimization = datetime.fromisoformat(history["last_optimization"])
            current_time = datetime.now()

            # 蜑榊屓譛驕ｩ蛹悶°繧臥ｵ碁℃譌･謨ｰ
            days_since_last = (current_time - last_optimization).days

            # 3譌･莉･荳顔ｵ碁℃縺励※縺・ｋ蝣ｴ蜷医・譛驕ｩ蛹門ｮ溯｡・
            if days_since_last >= 3:
                return True

            # 譛域屆譌･縺ｾ縺溘・驥第屆譌･縺ｧ蜑肴律莉･髯阪↓譛驕ｩ蛹悶＠縺ｦ縺・↑縺・ｴ蜷・
            weekday = current_time.weekday()
            if weekday in [0, 4]:  # 譛域屆繝ｻ驥第屆
                # 蜑榊屓譛驕ｩ蛹悶′譏ｨ譌･繧医ｊ蜑阪↑繧牙ｮ溯｡・
                if days_since_last >= 1:
                    return True

            return False

        except Exception as e:
            self.logger.warning(f"譛驕ｩ蛹門ｱ･豁ｴ遒ｺ隱阪お繝ｩ繝ｼ: {e}")
            # 繧ｨ繝ｩ繝ｼ譎ゅ・螳牙・縺ｮ縺溘ａ譛驕ｩ蛹門ｮ溯｡・
            return True  # 譛域屆繝ｻ驥第屆

    def _save_optimization_history(self, symbols: List[str]):
        """譛驕ｩ蛹門ｱ･豁ｴ繧剃ｿ晏ｭ・""
        try:
            history = {
                "last_optimization": datetime.now().isoformat(),
                "symbols": symbols,
                "symbol_count": len(symbols),
                "optimization_type": "tse4000_auto",
            }

            history_file = "tse4000_optimization_history.json"
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)

            self.logger.info(f"譛驕ｩ蛹門ｱ･豁ｴ菫晏ｭ・ {len(symbols)}驫俶氛")

        except Exception as e:
            self.logger.error(f"螻･豁ｴ菫晏ｭ倥お繝ｩ繝ｼ: {e}")

    def _load_previous_optimization(self) -> List[str]:
        """蜑榊屓縺ｮ譛驕ｩ蛹也ｵ先棡繧定ｪｭ縺ｿ霎ｼ縺ｿ"""
        try:
            history_file = "tse4000_optimization_history.json"

            if not os.path.exists(history_file):
                self.logger.info("螻･豁ｴ繝輔ぃ繧､繝ｫ縺ｪ縺励ゅョ繝輔か繝ｫ繝磯釜譟・ｽｿ逕ｨ")
                return self._get_default_symbols()

            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)

            if "symbols" in history and history["symbols"]:
                symbols = history["symbols"]
                self.logger.info(f"蜑榊屓譛驕ｩ蛹也ｵ先棡隱ｭ縺ｿ霎ｼ縺ｿ: {len(symbols)}驫俶氛")
                return symbols

        except Exception as e:
            self.logger.error(f"螻･豁ｴ隱ｭ縺ｿ霎ｼ縺ｿ繧ｨ繝ｩ繝ｼ: {e}")

        return self._get_default_symbols()

    def _get_default_symbols(self) -> List[str]:
        """繝・ヵ繧ｩ繝ｫ繝磯釜譟・Μ繧ｹ繝・""
        return [
            "6758.T",  # 繧ｽ繝九・繧ｰ繝ｫ繝ｼ繝・
            "7203.T",  # 繝医Κ繧ｿ閾ｪ蜍戊ｻ・
            "8306.T",  # 荳芽廠UFJ驫陦・
            "4502.T",  # 豁ｦ逕ｰ阮ｬ蜩∝ｷ･讌ｭ
            "9984.T",  # 繧ｽ繝輔ヨ繝舌Φ繧ｯ繧ｰ繝ｫ繝ｼ繝・
            "6861.T",  # 繧ｭ繝ｼ繧ｨ繝ｳ繧ｹ
            "7974.T",  # 莉ｻ螟ｩ蝣・
            "4689.T",  # 繝､繝輔・
            "8035.T",  # 譚ｱ莠ｬ繧ｨ繝ｬ繧ｯ繝医Ο繝ｳ
            "6098.T",  # 繝ｪ繧ｯ繝ｫ繝ｼ繝医・繝ｼ繝ｫ繝・ぅ繝ｳ繧ｰ繧ｹ
        ]

    async def _auto_learning_and_training(self, symbols: List[str]):
        """閾ｪ蜍募ｭｦ鄙偵・險鍋ｷｴ螳滓命"""
        self.logger.info("ｧ 蟄ｦ鄙偵・險鍋ｷｴ閾ｪ蜍募ｮ滓命荳ｭ...")

        try:
            # 荳ｦ蛻励〒蜷・釜譟・・蟄ｦ鄙貞ｮ溯｡・
            learning_tasks = []

            for symbol in symbols:
                task = self._learn_single_symbol(symbol)
                learning_tasks.append(task)

            # 荳ｦ蛻怜ｮ溯｡・
            learning_results = await asyncio.gather(
                *learning_tasks, return_exceptions=True
            )

            successful_learning = len(
                [r for r in learning_results if not isinstance(r, Exception)]
            )
            self.logger.info(f"笨・蟄ｦ鄙貞ｮ御ｺ・ {successful_learning}/{len(symbols)}驫俶氛")

        except Exception as e:
            self.logger.error(f"蟄ｦ鄙偵・險鍋ｷｴ繧ｨ繝ｩ繝ｼ: {str(e)}")

    async def _learn_single_symbol(self, symbol: str) -> Dict[str, Any]:
        """蜊倅ｸ驫俶氛蟄ｦ鄙・""
        try:
            # 萓｡譬ｼ繝・・繧ｿ蜿門ｾ・
            price_data = self.data_provider.get_stock_data(symbol)

            if price_data.empty:
                return {"symbol": symbol, "status": "no_data"}

            # 莠域ｸｬ螳溯｡鯉ｼ亥ｭｦ鄙貞柑譫懆ｾｼ縺ｿ・・ 髱槫酔譛溘〒螳溯｡・
            loop = asyncio.get_event_loop()
            prediction_result = await loop.run_in_executor(
                None,
                self.hybrid_predictor.predict,
                symbol,
                PredictionMode.RESEARCH_MODE,
            )

            # 螳滓凾髢灘ｭｦ鄙偵す繧ｹ繝・Β縺ｫ繝・・繧ｿ霑ｽ蜉
            if (
                hasattr(self.hybrid_predictor, "real_time_learning_enabled")
                and self.hybrid_predictor.real_time_learning_enabled
            ):
                market_data = {
                    "symbol": symbol,
                    "price": (
                        price_data["Close"].iloc[-1] if "Close" in price_data else 1000
                    ),
                    "volume": (
                        price_data["Volume"].iloc[-1]
                        if "Volume" in price_data
                        else 100000
                    ),
                    "timestamp": datetime.now(),
                }
                # 螳滓凾髢灘ｭｦ鄙偵ｂ髱槫酔譛溘〒螳溯｡・
                if hasattr(self.hybrid_predictor, "process_real_time_market_data"):
                    try:
                        await loop.run_in_executor(
                            None,
                            self.hybrid_predictor.process_real_time_market_data,
                            market_data,
                        )
                    except Exception as rt_error:
                        self.logger.warning(f"螳滓凾髢灘ｭｦ鄙偵お繝ｩ繝ｼ {symbol}: {rt_error}")

            return {
                "symbol": symbol,
                "status": "success",
                "prediction": (
                    prediction_result.prediction if prediction_result else None
                ),
            }

        except Exception as e:
            self.logger.error(f"驫俶氛{symbol}蟄ｦ鄙偵お繝ｩ繝ｼ: {str(e)}")
            return {"symbol": symbol, "status": "error", "error": str(e)}

    async def _generate_auto_recommendations(
        self, symbols: List[str]
    ) -> List[AutoRecommendation]:
        """閾ｪ蜍墓耳螂ｨ逕滓・"""
        self.logger.info("識 謚戊ｳ・耳螂ｨ閾ｪ蜍慕函謌蝉ｸｭ...")

        recommendations = []

        try:
            for symbol in symbols:
                recommendation = await self._analyze_single_symbol(symbol)
                if recommendation:
                    recommendations.append(recommendation)

            # 繝ｪ繧ｹ繧ｯ繝ｻ繝ｪ繧ｿ繝ｼ繝ｳ縺ｧ繧ｽ繝ｼ繝・
            recommendations.sort(key=lambda x: x.expected_return, reverse=True)

            self.logger.info(f"笨・謗ｨ螂ｨ逕滓・螳御ｺ・ {len(recommendations)}驫俶氛")
            return recommendations

        except Exception as e:
            self.logger.error(f"謗ｨ螂ｨ逕滓・繧ｨ繝ｩ繝ｼ: {str(e)}")
            return recommendations

    async def _analyze_single_symbol(self, symbol: str) -> Optional[AutoRecommendation]:
        """蜊倅ｸ驫俶氛蛻・梵"""
        try:
            # 萓｡譬ｼ繝・・繧ｿ蜿門ｾ・
            price_data = self.data_provider.get_stock_data(symbol)

            if price_data.empty:
                return None

            current_price = price_data["Close"].iloc[-1]

            # 1. 莠域ｸｬ螳溯｡・- 髱槫酔譛溘〒螳溯｡・
            loop = asyncio.get_event_loop()
            prediction_result = await loop.run_in_executor(
                None, self.hybrid_predictor.predict, symbol, PredictionMode.AUTO
            )

            if not prediction_result:
                return None

            # 2. 繧ｻ繝ｳ繝√Γ繝ｳ繝亥・譫・
            sentiment_result = self.sentiment_analyzer.analyze_comprehensive_sentiment(
                symbol=symbol, price_data=price_data
            )

            # 3. 謌ｦ逡･逕滓・
            strategies = self.strategy_generator.generate_comprehensive_strategy(
                symbol, price_data
            )
            signals = self.strategy_generator.generate_trading_signals(
                symbol,
                price_data,
                sentiment_data={
                    "current_sentiment": {"score": sentiment_result.sentiment_score}
                },
            )

            # 4. 繝ｪ繧ｹ繧ｯ蛻・梵
            portfolio_data = {"positions": {symbol: 100000}, "total_value": 100000}
            risk_analysis = self.risk_manager.analyze_portfolio_risk(
                portfolio_data, {symbol: price_data}
            )

            # 5. 螢ｲ雋ｷ繧ｿ繧､繝溘Φ繧ｰ險育ｮ・
            buy_timing, sell_timing = self._calculate_optimal_timing(
                prediction_result, sentiment_result, signals
            )

            # 6. 邱丞粋蛻､螳・
            if self._should_recommend(
                prediction_result, sentiment_result, risk_analysis
            ):
                return self._create_recommendation(
                    symbol,
                    current_price,
                    prediction_result,
                    sentiment_result,
                    buy_timing,
                    sell_timing,
                    risk_analysis,
                )

            return None

        except Exception as e:
            self.logger.error(f"驫俶氛{symbol}蛻・梵繧ｨ繝ｩ繝ｼ: {str(e)}")
            return None

    def _calculate_optimal_timing(
        self, prediction, sentiment, signals
    ) -> Tuple[datetime, datetime]:
        """譛驕ｩ螢ｲ雋ｷ繧ｿ繧､繝溘Φ繧ｰ險育ｮ・""
        current_time = datetime.now()

        # 雋ｷ縺・ち繧､繝溘Φ繧ｰ
        if sentiment.sentiment_score > 0.3 and prediction.confidence > 0.7:
            # 繝昴ず繝・ぅ繝悶↑蝣ｴ蜷医・譌ｩ繧√・繧ｨ繝ｳ繝医Μ繝ｼ
            buy_date = current_time + timedelta(days=1)
        elif sentiment.sentiment_score > 0:
            # 霆ｽ蠕ｮ繝昴ず繝・ぅ繝悶↑繧・-3譌･讒伜ｭ占ｦ・
            buy_date = current_time + timedelta(days=2)
        else:
            # 繝阪ぎ繝・ぅ繝悶↑繧・騾ｱ髢灘ｾ・ｩ・
            buy_date = current_time + timedelta(days=7)

        # 螢ｲ繧翫ち繧､繝溘Φ繧ｰ・域兜雉・悄髢薙・繝ｼ繧ｹ・・
        base_hold_period = self.auto_settings["investment_period_days"]

        # 菫｡鬆ｼ蠎ｦ縺ｫ繧医ｋ隱ｿ謨ｴ
        if prediction.confidence > 0.8:
            hold_period = base_hold_period + 10  # 鬮倅ｿ｡鬆ｼ蠎ｦ縺ｪ繧蛾聞譛滉ｿ晄怏
        elif prediction.confidence < 0.6:
            hold_period = base_hold_period - 10  # 菴惹ｿ｡鬆ｼ蠎ｦ縺ｪ繧画掠譛溷｣ｲ蜊ｴ
        else:
            hold_period = base_hold_period

        sell_date = buy_date + timedelta(days=hold_period)

        return buy_date, sell_date

    def _should_recommend(self, prediction, sentiment, risk_analysis) -> bool:
        """謗ｨ螂ｨ蛻､螳・""
        # 譛蟆丈ｿ｡鬆ｼ蠎ｦ繝√ぉ繝・け
        if prediction.confidence < self.auto_settings["min_confidence"]:
            return False

        # 繝ｪ繧ｹ繧ｯ繧ｹ繧ｳ繧｢繝√ぉ繝・け
        if risk_analysis.total_risk_score > self.auto_settings["max_risk_score"]:
            return False

        # 莠域ｸｬ萓｡譬ｼ荳頑・繝√ぉ繝・け
        if prediction.prediction <= 0:
            return False

        return True

    def _create_recommendation(
        self,
        symbol: str,
        current_price: float,
        prediction,
        sentiment,
        buy_timing: datetime,
        sell_timing: datetime,
        risk_analysis,
    ) -> AutoRecommendation:
        """謗ｨ螂ｨ諠・ｱ菴懈・"""

        # 逶ｮ讓吩ｾ｡譬ｼ・井ｺ域ｸｬ萓｡譬ｼ繝吶・繧ｹ・・
        target_price = prediction.prediction

        # 繧ｹ繝医ャ繝励Ο繧ｹ・・%荳具ｼ・
        stop_loss = current_price * 0.95

        # 譛溷ｾ・Μ繧ｿ繝ｼ繝ｳ險育ｮ・
        expected_return = (target_price - current_price) / current_price

        # 莨∵･ｭ蜷榊叙蠕暦ｼ育ｰ｡逡･蛹厄ｼ・
        company_name = self._get_company_name(symbol)

        # 謗ｨ螂ｨ逅・罰逕滓・
        reasoning = self._generate_reasoning(prediction, sentiment, risk_analysis)

        risk_score = getattr(risk_analysis, 'total_risk_score', 0.0)
        recommendation_score = self._calculate_recommendation_score(expected_return, prediction.confidence, risk_score)

        return AutoRecommendation(
            symbol=symbol,
            company_name=company_name,
            action=ActionType.BUY,
            entry_price=current_price,
            target_price=target_price,
            stop_loss=stop_loss,
            buy_date=buy_timing,
            sell_date=sell_timing,
            expected_return=expected_return,
            confidence=prediction.confidence,
            reasoning=reasoning,
            risk_level=risk_analysis.risk_level.value,
            recommendation_score=recommendation_score,
            risk_score=risk_score,
        )

    def _calculate_recommendation_score(self, expected_return: float, confidence: float, risk_score: float) -> float:
        expected_pct = max(min(expected_return * 100.0, 20.0), -20.0)
        expected_component = (expected_pct + 20.0) / 40.0  # 0..1
        confidence_component = max(0.0, min(confidence, 1.0))
        normalized_risk = max(0.0, min(risk_score / 4.0 if risk_score is not None else 0.0, 1.0))
        risk_component = 1.0 - normalized_risk
        score = (expected_component * 4.0) + (confidence_component * 4.0) + (risk_component * 2.0)
        return round(max(0.0, min(score, 10.0)), 2)

    def _ensure_models_ready(self) -> bool:
        ensemble_ready = False
        enhanced_system = getattr(self.hybrid_predictor, "enhanced_system", None)

        if enhanced_system:
            try:
                if enhanced_system.is_trained:
                    self.logger.info("譌｢蟄倥・繧｢繝ｳ繧ｵ繝ｳ繝悶Ν繝｢繝・Ν縺悟茜逕ｨ蜿ｯ閭ｽ縺ｧ縺・)
                    return True
                if enhanced_system.load_ensemble():
                    self.logger.info("菫晏ｭ俶ｸ医∩縺ｮ繧｢繝ｳ繧ｵ繝ｳ繝悶Ν繝｢繝・Ν繧定ｪｭ縺ｿ霎ｼ縺ｿ縺ｾ縺励◆")
                    return True
            except Exception as load_error:
                self.logger.warning(f"譌｢蟄倥Δ繝・Ν縺ｮ隱ｭ縺ｿ霎ｼ縺ｿ縺ｫ螟ｱ謨・ {load_error}")

        ensemble_file = Path('models/saved_models/ensemble_models.joblib')
        refresh_days = self.auto_settings.get("model_refresh_days", 30)
        file_stale = False

        if ensemble_file.exists():
            if refresh_days and refresh_days > 0:
                try:
                    file_age = datetime.now() - datetime.fromtimestamp(ensemble_file.stat().st_mtime)
                    if file_age >= timedelta(days=refresh_days):
                        file_stale = True
                        self.logger.info(
                            f"繧｢繝ｳ繧ｵ繝ｳ繝悶Ν繝｢繝・Ν縺鶏file_age.days}譌･髢捺峩譁ｰ縺輔ｌ縺ｦ縺・↑縺・◆繧∝・蟄ｦ鄙偵＠縺ｾ縺・
                        )
                except Exception as age_error:
                    self.logger.warning(f"繝｢繝・Ν繝輔ぃ繧､繝ｫ縺ｮ譖ｴ譁ｰ譌･譎ょ叙蠕励↓螟ｱ謨・ {age_error}")

            if enhanced_system and not file_stale:
                try:
                    if enhanced_system.load_ensemble():
                        self.logger.info("菫晏ｭ俶ｸ医∩縺ｮ繧｢繝ｳ繧ｵ繝ｳ繝悶Ν繝｢繝・Ν繧定ｪｭ縺ｿ霎ｼ縺ｿ縺ｾ縺励◆")
                        return True
                except Exception as e:
                    self.logger.warning(f"繝｢繝・Ν隱ｭ縺ｿ霎ｼ縺ｿ縺ｧ隴ｦ蜻・ {e}")

        self.logger.info("繧｢繝ｳ繧ｵ繝ｳ繝悶Ν繝｢繝・Ν縺瑚ｦ九▽縺九ｉ縺ｪ縺・°蜀榊ｭｦ鄙偵′蠢・ｦ√↑縺溘ａ閾ｪ蜍募ｭｦ鄙偵ｒ螳溯｡後＠縺ｾ縺・)

        import sys
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            except Exception as reconfig_error:
                self.logger.debug(f"stdout reconfigure skipped: {reconfig_error}")

        try:
            from train_enhanced_ensemble import (
                run_enhanced_ensemble_training,
                evaluate_trained_model,
            )
        except ImportError as e:
            self.logger.error(f"蟄ｦ鄙偵Δ繧ｸ繝･繝ｼ繝ｫ縺ｮ隱ｭ縺ｿ霎ｼ縺ｿ縺ｫ螟ｱ謨・ {e}")
            return False

        training_result = run_enhanced_ensemble_training()
        if not training_result.get('success'):
            self.logger.error(f"閾ｪ蜍募ｭｦ鄙偵↓螟ｱ謨・ {training_result.get('error', '逅・罰荳肴・')}")
            return False

        predictor = training_result.get('predictor')
        if predictor is not None:
            try:
                predictor.save_ensemble()
            except Exception as save_error:
                self.logger.warning(f"繧｢繝ｳ繧ｵ繝ｳ繝悶Ν菫晏ｭ倥↓螟ｱ謨・ {save_error}")
        else:
            self.logger.warning("蟄ｦ鄙堤ｵ先棡縺ｫ predictor 縺悟性縺ｾ繧後※縺・∪縺帙ｓ")

        try:
            evaluate_trained_model()
        except Exception as eval_error:
            self.logger.warning(f"繝｢繝・Ν隧穂ｾ｡縺ｧ隴ｦ蜻・ {eval_error}")

        if predictor is not None:
            try:
                predictor.is_trained = True
                self.hybrid_predictor.enhanced_system = predictor
                self.logger.info("閾ｪ蜍募ｭｦ鄙偵＠縺溘い繝ｳ繧ｵ繝ｳ繝悶Ν繝｢繝・Ν繧帝←逕ｨ縺励∪縺励◆")
                return True
            except Exception as attach_error:
                self.logger.warning(f"蟄ｦ鄙偵Δ繝・Ν縺ｮ驕ｩ逕ｨ縺ｫ螟ｱ謨・ {attach_error}")

        if enhanced_system:
            try:
                enhanced_system.is_trained = False
                if enhanced_system.load_ensemble():
                    self.logger.info("菫晏ｭ俶ｸ医∩繝｢繝・Ν繧貞・隱ｭ縺ｿ霎ｼ縺ｿ縺励∪縺励◆")
                    return True
            except Exception as reload_error:
                self.logger.warning(f"蜀崎ｪｭ縺ｿ霎ｼ縺ｿ縺ｫ螟ｱ謨・ {reload_error}")

        if ensemble_file.exists():
            self.logger.info("繧｢繝ｳ繧ｵ繝ｳ繝悶Ν繝｢繝・Ν繝輔ぃ繧､繝ｫ縺ｯ蟄伜惠縺励∪縺吶′隱ｭ縺ｿ霎ｼ縺ｿ縺ｫ螟ｱ謨励＠縺ｾ縺励◆")
            return True

        return False

    def _get_company_name(self, symbol: str) -> str:
        """莨∵･ｭ蜷榊叙蠕・""
        company_map = {
            "6758.T": "繧ｽ繝九・繧ｰ繝ｫ繝ｼ繝・,
            "7203.T": "繝医Κ繧ｿ閾ｪ蜍戊ｻ・,
            "8306.T": "荳芽廠UFJ驫陦・,
            "4502.T": "豁ｦ逕ｰ阮ｬ蜩∝ｷ･讌ｭ",
            "9984.T": "繧ｽ繝輔ヨ繝舌Φ繧ｯ繧ｰ繝ｫ繝ｼ繝・,
            "6861.T": "繧ｭ繝ｼ繧ｨ繝ｳ繧ｹ",
            "7974.T": "莉ｻ螟ｩ蝣・,
            "4689.T": "繝､繝輔・",
            "8035.T": "譚ｱ莠ｬ繧ｨ繝ｬ繧ｯ繝医Ο繝ｳ",
            "6098.T": "繝ｪ繧ｯ繝ｫ繝ｼ繝医・繝ｼ繝ｫ繝・ぅ繝ｳ繧ｰ繧ｹ",
        }
        return company_map.get(symbol, symbol)

    def _generate_reasoning(self, prediction, sentiment, risk_analysis) -> str:
        """謗ｨ螂ｨ逅・罰逕滓・"""
        reasons = []

        # 莠域ｸｬ繝吶・繧ｹ縺ｮ逅・罰
        if prediction.confidence > 0.8:
            reasons.append(f"鬮倅ｿ｡鬆ｼ蠎ｦ莠域ｸｬ({prediction.confidence:.1%})")

        # 繧ｻ繝ｳ繝√Γ繝ｳ繝医・繝ｼ繧ｹ縺ｮ逅・罰
        if sentiment.sentiment_score > 0.5:
            reasons.append("蠑ｷ縺・・繧ｸ繝・ぅ繝悶そ繝ｳ繝√Γ繝ｳ繝・)
        elif sentiment.sentiment_score > 0.2:
            reasons.append("繝昴ず繝・ぅ繝悶そ繝ｳ繝√Γ繝ｳ繝・)

        # 繝ｪ繧ｹ繧ｯ繝吶・繧ｹ縺ｮ逅・罰
        if risk_analysis.risk_level.value == "low":
            reasons.append("菴弱Μ繧ｹ繧ｯ")
        elif risk_analysis.risk_level.value == "medium":
            reasons.append("荳ｭ遞句ｺｦ繝ｪ繧ｹ繧ｯ")

        if not reasons:
            reasons.append("邱丞粋逧・愛譁ｭ縺ｫ繧医ｊ謗ｨ螂ｨ")

        return " + ".join(reasons)

    def _display_recommendations(self, recommendations: List[AutoRecommendation]):
        """謗ｨ螂ｨ邨先棡陦ｨ遉ｺ"""
        if not recommendations:
            print("\n[邨先棡] 迴ｾ蝨ｨ謗ｨ螂ｨ縺ｧ縺阪ｋ驫俶氛縺後≠繧翫∪縺帙ｓ")
            return

        print(f"\n[邨先棡] 繝輔Ν繧ｪ繝ｼ繝域兜雉・耳螂ｨ ({len(recommendations)}驫俶氛)")
        print("=" * 80)

        for i, rec in enumerate(recommendations, 1):
            print(f"\n縲先耳螂ｨ #{i}縲捜rec.company_name} ({rec.symbol})")
            print(f"  雋ｷ縺・ｾ｡譬ｼ: ﾂ･{rec.entry_price:,.0f}")
            print(f"  逶ｮ讓吩ｾ｡譬ｼ: ﾂ･{rec.target_price:,.0f}")
            print(f"  繧ｹ繝医ャ繝励Ο繧ｹ: ﾂ･{rec.stop_loss:,.0f}")
            print(f"  雋ｷ縺・凾譛・ {rec.buy_date.strftime('%Y蟷ｴ%m譛・d譌･ %H譎る・)}")
            print(f"  螢ｲ繧頑凾譛・ {rec.sell_date.strftime('%Y蟷ｴ%m譛・d譌･ %H譎る・)}")
            print(f"  譛溷ｾ・Μ繧ｿ繝ｼ繝ｳ: {rec.expected_return:.1%}")
            print(f"  菫｡鬆ｼ蠎ｦ: {rec.confidence:.1%}")
            print(f"  繝ｪ繧ｹ繧ｯ繝ｬ繝吶Ν: {rec.risk_level}")
            print(f"  逅・罰: {rec.reasoning}")

        print("\n" + "=" * 80)
        print("[豕ｨ諢従 縺薙ｌ繧峨・莠域ｸｬ縺ｫ蝓ｺ縺･縺乗耳螂ｨ縺ｧ縺ゅｊ縲∵兜雉・・閾ｪ蟾ｱ雋ｬ莉ｻ縺ｧ陦後▲縺ｦ縺上□縺輔＞")

    def get_system_status(self) -> Dict[str, Any]:
        """繧ｷ繧ｹ繝・Β迥ｶ豕∝叙蠕・""
        return {
            "tse_optimization_ready": True,
            "hybrid_predictor_ready": True,
            "sentiment_analyzer_ready": True,
            "strategy_generator_ready": True,
            "risk_manager_ready": True,
            "auto_settings": self.auto_settings,
            "last_run": datetime.now(),
        }


# 繝｡繧､繝ｳ螳溯｡碁未謨ｰ
async def run_full_auto():
    """繝輔Ν繧ｪ繝ｼ繝亥ｮ溯｡・""
    system = FullAutoInvestmentSystem()
    recommendations = await system.run_full_auto_analysis()
    return recommendations


if __name__ == "__main__":
    # 繝・せ繝亥ｮ溯｡・
    asyncio.run(run_full_auto())
