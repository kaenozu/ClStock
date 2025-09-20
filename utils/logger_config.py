#!/usr/bin/env python3
"""
ログ設定の一元管理モジュール
logging.basicConfigの複数回呼び出し問題を解決
"""

import logging
import sys
from pathlib import Path


def setup_logger(name: str = None, level: int = logging.INFO,
                format_string: str = None) -> logging.Logger:
    """
    安全なログ設定

    Args:
        name: ロガー名（Noneの場合はルートロガー）
        level: ログレベル
        format_string: フォーマット文字列

    Returns:
        設定済みロガー
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # ロガーを取得
    logger = logging.getLogger(name)

    # 既にハンドラーが設定されている場合は重複を避ける
    if logger.handlers:
        return logger

    # レベル設定
    logger.setLevel(level)

    # コンソールハンドラーを作成
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # フォーマッターを作成
    formatter = logging.Formatter(format_string)
    console_handler.setFormatter(formatter)

    # ハンドラーをロガーに追加
    logger.addHandler(console_handler)

    # 親ロガーへの伝播を防ぐ（重複ログを避ける）
    if name:
        logger.propagate = False

    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    ロガーを取得（既存の場合はそのまま返す）

    Args:
        name: ロガー名

    Returns:
        ロガー
    """
    logger = logging.getLogger(name)

    # まだ設定されていない場合は設定
    if not logger.handlers:
        return setup_logger(name)

    return logger


def set_log_level(level: int, logger_name: str = None):
    """
    安全にログレベルを変更

    Args:
        level: 新しいログレベル
        logger_name: ロガー名（Noneの場合はルートロガー）
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # ハンドラーのレベルも更新
    for handler in logger.handlers:
        handler.setLevel(level)