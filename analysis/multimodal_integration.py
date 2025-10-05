"""マルチモーダルデータ統合処理"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class MultimodalIntegrator:
    """株価・ファンダメンタルズ・センチメントのマルチモーダルデータ統合"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.price_scaler = StandardScaler()
        self.fundamental_scaler = StandardScaler()
        self.sentiment_scaler = StandardScaler()
        
    def integrate_data(self, price_data, fundamentals_data=None, sentiment_data=None):
        """
        株価、ファンダメンタルズ、センチメントデータを統合
        
        Args:
            price_data: 株価データ (pandas DataFrame)
            fundamentals_data: ファンダメンタルデータ (pandas DataFrame, optional)
            sentiment_data: センチメントデータ (pandas DataFrame, optional)
            
        Returns:
            numpy array: 統合された特徴量
        """
        # 株価データの前処理
        price_features = self._process_price_data(price_data)
        
        # ファンダメンタルズデータの前処理
        fundamental_features = self._process_fundamental_data(fundamentals_data) if fundamentals_data is not None else np.array([])
        
        # センチメントデータの前処理
        sentiment_features = self._process_sentiment_data(sentiment_data) if sentiment_data is not None else np.array([])
        
        # 統合
        all_features = [price_features]
        if fundamental_features.size > 0:
            all_features.append(fundamental_features)
        if sentiment_features.size > 0:
            all_features.append(sentiment_features)
            
        # サイズを合わせるための調整
        min_length = min([f.shape[0] for f in all_features])
        aligned_features = []
        for f in all_features:
            if f.shape[0] > min_length:
                aligned_features.append(f[:min_length])
            elif f.shape[0] < min_length:
                # 短い場合はゼロパディング
                padding = np.zeros((min_length - f.shape[0], f.shape[1]))
                aligned_features.append(np.vstack([f, padding]))
            else:
                aligned_features.append(f)
                
        # 統合された特徴量を返す
        integrated_features = np.concatenate(aligned_features, axis=1)
        
        # 全体のスケーリング
        integrated_features_scaled = self.scaler.fit_transform(integrated_features)
        
        logger.info(f"マルチモーダルデータ統合完了: {integrated_features_scaled.shape}")
        return integrated_features_scaled
        
    def _process_price_data(self, price_data):
        """株価データの前処理"""
        # OHLCVなどの特徴量を抽出
        features = []
        
        # 各カラムを正規化
        if 'Close' in price_data.columns:
            close_prices = price_data['Close'].values.reshape(-1, 1)
            scaled_close = self.price_scaler.fit_transform(close_prices)
            features.append(scaled_close)
            
        if 'Volume' in price_data.columns:
            volumes = price_data['Volume'].values.reshape(-1, 1)
            scaled_volume = self.price_scaler.fit_transform(volumes)
            features.append(scaled_volume)
            
        if 'Open' in price_data.columns:
            opens = price_data['Open'].values.reshape(-1, 1)
            scaled_open = self.price_scaler.fit_transform(opens)
            features.append(scaled_open)
            
        if 'High' in price_data.columns:
            highs = price_data['High'].values.reshape(-1, 1)
            scaled_high = self.price_scaler.fit_transform(highs)
            features.append(scaled_high)
            
        if 'Low' in price_data.columns:
            lows = price_data['Low'].values.reshape(-1, 1)
            scaled_low = self.price_scaler.fit_transform(lows)
            features.append(scaled_low)
            
        return np.hstack(features) if len(features) > 1 else features[0]
        
    def _process_fundamental_data(self, fundamentals_data):
        """ファンダメンタルズデータの前処理"""
        if fundamentals_data is None or fundamentals_data.empty:
            return np.array([]).reshape(0, 0)
            
        # ファンダメンタル指標のスケーリング
        scaled_fundamentals = self.fundamental_scaler.fit_transform(fundamentals_data.values)
        
        # 欠損値を平均で埋める
        scaled_fundamentals = np.nan_to_num(scaled_fundamentals, nan=np.nanmean(scaled_fundamentals))
        
        return scaled_fundamentals
        
    def _process_sentiment_data(self, sentiment_data):
        """センチメントデータの前処理"""
        if sentiment_data is None or sentiment_data.empty:
            return np.array([]).reshape(0, 0)
            
        # センチメントスコアのスケーリング
        scaled_sentiment = self.sentiment_scaler.fit_transform(sentiment_data.values)
        
        # 欠損値を中央値で埋める
        scaled_sentiment = np.nan_to_num(scaled_sentiment, nan=np.nanmedian(scaled_sentiment))
        
        return scaled_sentiment