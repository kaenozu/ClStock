"""カスタムアテンション機構（Temporal Attention）"""
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TemporalAttention(Layer):
    """時系列データ用カスタムアテンション機構"""
    
    def __init__(self, attention_units, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)
        self.attention_units = attention_units
        self.W_query = Dense(attention_units, activation='tanh')
        self.W_key = Dense(attention_units, activation='tanh')
        self.W_value = Dense(attention_units, activation='tanh')
        self.V = Dense(1)
        
    def build(self, input_shape):
        super().build(input_shape)
        
    def call(self, inputs):
        """
        Args:
            inputs: 時系列データ (batch_size, time_steps, features)
            
        Returns:
            attended_output: アテンション適用後の出力 (batch_size, features)
        """
        # Query, Key, Valueの計算
        query = self.W_query(inputs)  # (batch_size, time_steps, attention_units)
        key = self.W_key(inputs)      # (batch_size, time_steps, attention_units)
        value = self.W_value(inputs)  # (batch_size, time_steps, attention_units)
        
        # アテンションスコアの計算
        scores = self.V(tf.nn.tanh(query + key))  # (batch_size, time_steps, 1)
        attention_weights = tf.nn.softmax(scores, axis=1)  # (batch_size, time_steps, 1)
        
        # アテンション重みを用いた加重平均
        attended_output = tf.reduce_sum(attention_weights * value, axis=1)  # (batch_size, attention_units)
        
        return attended_output

class MultiHeadTemporalAttention(Layer):
    """Multi-Head Temporal Attention"""
    
    def __init__(self, num_heads, attention_units_per_head, **kwargs):
        super(MultiHeadTemporalAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention_units_per_head = attention_units_per_head
        self.total_units = num_heads * attention_units_per_head
        
        self.attention_layers = [
            TemporalAttention(attention_units_per_head) 
            for _ in range(num_heads)
        ]
        self.output_projection = Dense(self.total_units)
        
    def call(self, inputs):
        """
        Args:
            inputs: 時系列データ (batch_size, time_steps, features)
            
        Returns:
            multi_head_output: Multi-Headアテンション適用後の出力 (batch_size, total_units)
        """
        head_outputs = []
        for attention_layer in self.attention_layers:
            head_output = attention_layer(inputs)
            head_outputs.append(head_output)
            
        # 複数のヘッドの出力を連結
        concatenated_heads = tf.concat(head_outputs, axis=-1)
        
        # 出力プロジェクション
        multi_head_output = self.output_projection(concatenated_heads)
        
        return multi_head_output

class AdaptiveTemporalAttention(Layer):
    """適応的時系列アテンション（短期・中期・長期の重み調整可能）"""
    
    def __init__(self, attention_units, **kwargs):
        super(AdaptiveTemporalAttention, self).__init__(**kwargs)
        self.attention_units = attention_units
        self.short_term_attention = TemporalAttention(attention_units // 3)
        self.medium_term_attention = TemporalAttention(attention_units // 3)
        self.long_term_attention = TemporalAttention(attention_units // 3)
        self.adaptive_weights = Dense(3, activation='softmax')  # 短期・中期・長期の重み
        self.output_projection = Dense(attention_units)
        
    def call(self, inputs):
        """
        Args:
            inputs: 時系列データ (batch_size, time_steps, features)
            
        Returns:
            adaptive_output: 適応的アテンション適用後の出力 (batch_size, attention_units)
        """
        seq_len = tf.shape(inputs)[1]
        
        # 短期アテンション（最近のデータ）
        short_inputs = inputs[:, -seq_len//3:, :]
        short_attention = self.short_term_attention(short_inputs)
        
        # 中期アテンション（中間のデータ）
        mid_start = seq_len // 3
        mid_end = 2 * seq_len // 3
        mid_inputs = inputs[:, mid_start:mid_end, :]
        mid_attention = self.medium_term_attention(mid_inputs)
        
        # 長期アテンション（古いデータ）
        long_inputs = inputs[:, :seq_len//3, :]
        long_attention = self.long_term_attention(long_inputs)
        
        # 全アテンションの連結
        all_attentions = tf.stack([short_attention, mid_attention, long_attention], axis=1)
        
        # 適応的重みの計算
        weights = self.adaptive_weights(tf.reduce_mean(all_attentions, axis=1, keepdims=True))
        weights = tf.squeeze(weights, axis=1)
        
        # 重み付き平均
        adaptive_output = (
            weights[:, 0:1] * short_attention +
            weights[:, 1:2] * mid_attention +
            weights[:, 2:3] * long_attention
        )
        
        # 出力プロジェクション
        adaptive_output = self.output_projection(adaptive_output)
        
        return adaptive_output