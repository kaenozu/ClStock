#!/usr/bin/env python3
"""
セクター分類の一元管理
Single Source of Truth for sector classification
"""

class SectorClassification:
    """セクター分類の一元管理クラス"""

    # セクター別銘柄リスト（Single Source of Truth）
    SECTORS = {
        'auto': {
            'name': '自動車・輸送機器',
            'risk_level': 0.4,
            'risk_description': '景気敏感',
            'symbols': ['7203.T', '7267.T', '7201.T', '7269.T', '7211.T']
        },
        'tech': {
            'name': '電機・精密機器',
            'risk_level': 0.35,
            'risk_description': '技術変化リスク',
            'symbols': ['6758.T', '6861.T', '6954.T', '6981.T', '6503.T', '6702.T', '6752.T', '6971.T', '7751.T', '7832.T']
        },
        'comm': {
            'name': '通信・IT',
            'risk_level': 0.4,
            'risk_description': '競争激化',
            'symbols': ['9984.T', '9433.T', '9434.T', '6098.T', '4385.T']
        },
        'finance': {
            'name': '金融',
            'risk_level': 0.3,
            'risk_description': '金利リスク',
            'symbols': ['8306.T', '8316.T', '8411.T', '8604.T', '8473.T']
        },
        'trading': {
            'name': '商社',
            'risk_level': 0.35,
            'risk_description': '資源価格リスク',
            'symbols': ['8001.T', '8058.T', '8031.T', '8053.T', '8002.T']
        },
        'pharma': {
            'name': '医薬品・化学',
            'risk_level': 0.25,
            'risk_description': '規制リスク',
            'symbols': ['4502.T', '4507.T', '4503.T', '4005.T', '4063.T', '4183.T']
        },
        'material': {
            'name': '素材・エネルギー',
            'risk_level': 0.45,
            'risk_description': '商品価格リスク',
            'symbols': ['5401.T', '5713.T', '1605.T', '5020.T']
        },
        'consumer': {
            'name': '消費・小売',
            'risk_level': 0.3,
            'risk_description': '消費動向リスク',
            'symbols': ['7974.T', '8267.T', '9983.T', '3382.T', '2914.T']
        },
        'real_estate': {
            'name': '不動産・建設',
            'risk_level': 0.35,
            'risk_description': '金利・景気リスク',
            'symbols': ['8802.T', '8801.T', '1801.T', '6367.T']
        }
    }

    DEFAULT_RISK_LEVEL = 0.3  # デフォルトリスクレベル

    @classmethod
    def get_sector_risk(cls, symbol: str) -> float:
        """銘柄のセクターリスクを取得"""
        for sector_data in cls.SECTORS.values():
            if symbol in sector_data['symbols']:
                return sector_data['risk_level']
        return cls.DEFAULT_RISK_LEVEL

    @classmethod
    def get_sector_name(cls, symbol: str) -> str:
        """銘柄のセクター名を取得"""
        for sector_data in cls.SECTORS.values():
            if symbol in sector_data['symbols']:
                return sector_data['name']
        return 'その他'

    @classmethod
    def get_sector_info(cls, symbol: str) -> dict:
        """銘柄の完全なセクター情報を取得"""
        for sector_key, sector_data in cls.SECTORS.items():
            if symbol in sector_data['symbols']:
                return {
                    'sector_key': sector_key,
                    'name': sector_data['name'],
                    'risk_level': sector_data['risk_level'],
                    'risk_description': sector_data['risk_description']
                }
        return {
            'sector_key': 'other',
            'name': 'その他',
            'risk_level': cls.DEFAULT_RISK_LEVEL,
            'risk_description': 'デフォルトリスク'
        }

    @classmethod
    def get_all_symbols_by_sector(cls, sector_key: str) -> list:
        """指定セクターの全銘柄を取得"""
        if sector_key in cls.SECTORS:
            return cls.SECTORS[sector_key]['symbols'].copy()
        return []

    @classmethod
    def get_sector_distribution(cls) -> dict:
        """セクター別銘柄数分布を取得"""
        distribution = {}
        for sector_key, sector_data in cls.SECTORS.items():
            distribution[sector_data['name']] = len(sector_data['symbols'])
        return distribution