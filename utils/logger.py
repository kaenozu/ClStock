import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logging(
    log_file="app.log",
    level=logging.INFO,
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
):
    """アプリケーションのロギングを設定する。

    Args:
        log_file (str): ログファイルのパス。
        level (int): ロギングレベル (例: logging.INFO, logging.DEBUG)。
        max_bytes (int): ログファイルの最大サイズ (バイト単位)。
        backup_count (int): 保持するバックアップログファイルの数。

    """
    # ログディレクトリが存在しない場合は作成
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # ルートロガーを取得
    logger = logging.getLogger()
    logger.setLevel(level)

    # 既存のハンドラをクリア
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)

    # フォーマッターの定義
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # コンソールハンドラの設定
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ファイルハンドラの設定 (ローテーション機能付き)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(
        f"Logging setup complete. Log file: {log_file}, Level: {logging.getLevelName(level)}",
    )
