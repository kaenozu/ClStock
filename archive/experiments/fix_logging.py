#!/usr/bin/env python3
"""
logging.basicConfig問題の一括修正スクリプト
"""

import os
import re
from pathlib import Path

def fix_logging_in_file(file_path: Path):
    """単一ファイルのログ設定を修正"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # logging.basicConfig の行を探す
        if '
            # import logging の後に utils.logger_config を追加
            if 'from utils.logger_config import setup_logger' not in content:
                content = re.sub(
                    r'(import logging\n)',
                    r'\1from utils.logger_config import setup_logger\n',
                    content
                )

            # logging.basicConfig(...) を削除し、logger を設定
            content = re.sub(
                r'logging\.basicConfig\([^)]*\)\n',
                '',
                content
            )

            # logger = setup_logger(__name__) を置換
            if 'logger = setup_logger(__name__)' in content:
                content = content.replace(
                    'logger = setup_logger(__name__)',
                    'logger = setup_logger(__name__)'
                )
            else:
                # logger が定義されていない場合は追加
                content = re.sub(
                    r'(from utils\.logger_config import setup_logger\n)',
                    r'\1logger = setup_logger(__name__)\n',
                    content
                )

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            return True
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False

    return False

def main():
    """メイン実行"""
    root_dir = Path('.')
    files_fixed = 0

    # Python ファイルを検索
    for py_file in root_dir.rglob('*.py'):
        if fix_logging_in_file(py_file):
            files_fixed += 1

    print(f"Fixed {files_fixed} files")

if __name__ == "__main__":
    main()