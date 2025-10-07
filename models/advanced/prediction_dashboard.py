#!/usr/bin/env python3
"""予測可視化ダッシュボード
リアルタイム予測結果を美しく可視化するWebダッシュボード
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import dash
    import dash_bootstrap_components as dbc
    from dash import Input, Output, dcc, html

    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False


@dataclass
class VisualizationData:
    """可視化データ構造"""

    symbol: str
    predictions: List[Dict[str, Any]]
    historical_data: pd.DataFrame
    sentiment_data: Optional[Dict[str, Any]]
    performance_metrics: Dict[str, float]
    timestamp: datetime


class ChartGenerator:
    """チャート生成器"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # カラーパレット
        self.colors = {
            "primary": "#1f77b4",
            "success": "#2ca02c",
            "warning": "#ff7f0e",
            "danger": "#d62728",
            "info": "#17a2b8",
            "light": "#f8f9fa",
            "dark": "#343a40",
        }

        # チャートテーマ設定
        if PLOTLY_AVAILABLE:
            pio.templates.default = "plotly_white"

    def create_prediction_chart(
        self,
        symbol: str,
        historical_data: pd.DataFrame,
        predictions: List[Dict],
        title: str = None,
    ) -> str:
        """予測チャート作成"""
        if not PLOTLY_AVAILABLE:
            return "<div>Plotly not available</div>"

        try:
            fig = make_subplots(
                rows=2,
                cols=1,
                row_heights=[0.7, 0.3],
                subplot_titles=(f"{symbol} 価格予測", "出来高"),
                vertical_spacing=0.1,
            )

            # 履歴価格データ
            if not historical_data.empty:
                fig.add_trace(
                    go.Candlestick(
                        x=historical_data.index,
                        open=historical_data.get("Open", historical_data["Close"]),
                        high=historical_data.get("High", historical_data["Close"]),
                        low=historical_data.get("Low", historical_data["Close"]),
                        close=historical_data["Close"],
                        name="実際価格",
                        increasing_line_color=self.colors["success"],
                        decreasing_line_color=self.colors["danger"],
                    ),
                    row=1,
                    col=1,
                )

                # 移動平均線
                if len(historical_data) >= 20:
                    ma20 = historical_data["Close"].rolling(window=20).mean()
                    fig.add_trace(
                        go.Scatter(
                            x=historical_data.index,
                            y=ma20,
                            mode="lines",
                            name="MA20",
                            line=dict(color=self.colors["warning"], width=1),
                        ),
                        row=1,
                        col=1,
                    )

            # 予測データ
            if predictions:
                pred_times = []
                pred_values = []
                confidence_upper = []
                confidence_lower = []

                for pred in predictions:
                    if "timestamp" in pred and "prediction" in pred:
                        pred_times.append(pred["timestamp"])
                        pred_values.append(pred["prediction"])

                        confidence = pred.get("confidence", 0.8)
                        pred_val = pred["prediction"]
                        margin = pred_val * (1 - confidence) * 0.1

                        confidence_upper.append(pred_val + margin)
                        confidence_lower.append(pred_val - margin)

                if pred_times:
                    # 予測ライン
                    fig.add_trace(
                        go.Scatter(
                            x=pred_times,
                            y=pred_values,
                            mode="lines+markers",
                            name="予測価格",
                            line=dict(color=self.colors["primary"], width=3),
                            marker=dict(size=8),
                        ),
                        row=1,
                        col=1,
                    )

                    # 信頼区間
                    fig.add_trace(
                        go.Scatter(
                            x=pred_times + pred_times[::-1],
                            y=confidence_upper + confidence_lower[::-1],
                            fill="toself",
                            fillcolor=f"rgba{tuple(list(bytes.fromhex(self.colors['primary'][1:])) + [0.2])}",
                            line=dict(color="rgba(255,255,255,0)"),
                            name="信頼区間",
                            showlegend=True,
                        ),
                        row=1,
                        col=1,
                    )

            # 出来高
            if not historical_data.empty and "Volume" in historical_data.columns:
                colors_volume = [
                    "red" if close < open else "green"
                    for close, open in zip(
                        historical_data["Close"],
                        historical_data.get("Open", historical_data["Close"]),
                    )
                ]

                fig.add_trace(
                    go.Bar(
                        x=historical_data.index,
                        y=historical_data["Volume"],
                        name="出来高",
                        marker_color=colors_volume,
                        opacity=0.7,
                    ),
                    row=2,
                    col=1,
                )

            # レイアウト設定
            fig.update_layout(
                title=title or f"{symbol} 予測分析チャート",
                xaxis_rangeslider_visible=False,
                height=800,
                showlegend=True,
                template="plotly_white",
            )

            return fig.to_html(include_plotlyjs="cdn")

        except Exception as e:
            self.logger.error(f"Chart creation failed: {e!s}")
            return f"<div>チャート生成エラー: {e!s}</div>"

    def create_sentiment_chart(self, sentiment_data: Dict[str, Any]) -> str:
        """センチメントチャート作成"""
        if not PLOTLY_AVAILABLE or not sentiment_data:
            return "<div>センチメントデータなし</div>"

        try:
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "センチメントスコア",
                    "ソース別分析",
                    "トレンド",
                    "モメンタム",
                ),
                specs=[
                    [{"type": "indicator"}, {"type": "bar"}],
                    [{"type": "scatter"}, {"type": "indicator"}],
                ],
            )

            # センチメントスコア（ゲージ）
            score = sentiment_data.get("current_sentiment", {}).get("score", 0)
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=score,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "センチメント"},
                    gauge={
                        "axis": {"range": [-1, 1]},
                        "bar": {"color": self.colors["primary"]},
                        "steps": [
                            {"range": [-1, -0.3], "color": self.colors["danger"]},
                            {"range": [-0.3, 0.3], "color": self.colors["warning"]},
                            {"range": [0.3, 1], "color": self.colors["success"]},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 0,
                        },
                    },
                ),
                row=1,
                col=1,
            )

            # ソース別分析
            sources = sentiment_data.get("sources_breakdown", {})
            if sources:
                source_names = list(sources.keys())
                source_values = list(sources.values())

                fig.add_trace(
                    go.Bar(
                        x=source_names,
                        y=source_values,
                        name="ソース別",
                        marker_color=[
                            self.colors["success"] if v > 0 else self.colors["danger"]
                            for v in source_values
                        ],
                    ),
                    row=1,
                    col=2,
                )

            # トレンド（時系列がある場合）
            trend_data = sentiment_data.get("trend", {})
            if "recent_sentiments" in trend_data:
                recent = trend_data["recent_sentiments"]
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(recent))),
                        y=recent,
                        mode="lines+markers",
                        name="最近のトレンド",
                        line=dict(color=self.colors["info"]),
                    ),
                    row=2,
                    col=1,
                )

            # モメンタム
            momentum = sentiment_data.get("current_sentiment", {}).get("momentum", 0)
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=momentum,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "モメンタム"},
                    gauge={
                        "axis": {"range": [-1, 1]},
                        "bar": {"color": self.colors["info"]},
                        "steps": [
                            {"range": [-1, -0.3], "color": "lightgray"},
                            {"range": [-0.3, 0.3], "color": "gray"},
                            {"range": [0.3, 1], "color": "lightblue"},
                        ],
                    },
                ),
                row=2,
                col=2,
            )

            fig.update_layout(
                title="センチメント分析ダッシュボード", height=600, showlegend=False,
            )

            return fig.to_html(include_plotlyjs="cdn")

        except Exception as e:
            self.logger.error(f"Sentiment chart creation failed: {e!s}")
            return f"<div>センチメントチャート生成エラー: {e!s}</div>"

    def create_performance_chart(self, performance_metrics: Dict[str, float]) -> str:
        """パフォーマンスチャート作成"""
        if not PLOTLY_AVAILABLE or not performance_metrics:
            return "<div>パフォーマンスデータなし</div>"

        try:
            # メトリクス表示
            metrics_html = "<div style='display: flex; flex-wrap: wrap; gap: 20px;'>"

            for metric_name, value in performance_metrics.items():
                if isinstance(value, (int, float)):
                    color = (
                        self.colors["success"] if value > 0 else self.colors["danger"]
                    )
                    metrics_html += f"""
                    <div style='border: 1px solid #ddd; padding: 15px; border-radius: 8px; text-align: center;'>
                        <h4 style='margin: 0; color: {color};'>{value:.3f}</h4>
                        <p style='margin: 5px 0 0 0; font-size: 14px;'>{metric_name}</p>
                    </div>
                    """

            metrics_html += "</div>"
            return metrics_html

        except Exception as e:
            self.logger.error(f"Performance chart creation failed: {e!s}")
            return f"<div>パフォーマンスチャート生成エラー: {e!s}</div>"


class DashboardGenerator:
    """ダッシュボード生成器"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.chart_generator = ChartGenerator()

    def generate_static_dashboard(self, visualization_data: VisualizationData) -> str:
        """静的ダッシュボード生成"""
        try:
            # 予測チャート
            prediction_chart = self.chart_generator.create_prediction_chart(
                visualization_data.symbol,
                visualization_data.historical_data,
                visualization_data.predictions,
            )

            # センチメントチャート
            sentiment_chart = ""
            if visualization_data.sentiment_data:
                sentiment_chart = self.chart_generator.create_sentiment_chart(
                    visualization_data.sentiment_data,
                )

            # パフォーマンスチャート
            performance_chart = self.chart_generator.create_performance_chart(
                visualization_data.performance_metrics,
            )

            # HTMLテンプレート
            html_template = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{visualization_data.symbol} 予測ダッシュボード</title>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                    body {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        margin: 0;
                        padding: 20px;
                        background-color: #f8f9fa;
                    }}
                    .header {{
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 30px;
                        border-radius: 10px;
                        margin-bottom: 30px;
                        text-align: center;
                    }}
                    .section {{
                        background: white;
                        padding: 20px;
                        margin-bottom: 20px;
                        border-radius: 10px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    }}
                    .grid {{
                        display: grid;
                        grid-template-columns: 2fr 1fr;
                        gap: 20px;
                    }}
                    @media (max-width: 768px) {{
                        .grid {{
                            grid-template-columns: 1fr;
                        }}
                    }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>{visualization_data.symbol} 予測ダッシュボード</h1>
                    <p>最終更新: {visualization_data.timestamp.strftime("%Y-%m-%d %H:%M:%S")}</p>
                </div>

                <div class="grid">
                    <div>
                        <div class="section">
                            <h2>価格予測チャート</h2>
                            {prediction_chart}
                        </div>

                        {f'<div class="section"><h2>センチメント分析</h2>{sentiment_chart}</div>' if sentiment_chart else ""}
                    </div>

                    <div>
                        <div class="section">
                            <h2>パフォーマンス指標</h2>
                            {performance_chart}
                        </div>

                        <div class="section">
                            <h2>予測サマリー</h2>
                            {self._generate_prediction_summary(visualization_data)}
                        </div>
                    </div>
                </div>

                <script>
                    // 自動リフレッシュ（30秒ごと）
                    setTimeout(function() {{
                        window.location.reload();
                    }}, 30000);
                </script>
            </body>
            </html>
            """

            return html_template

        except Exception as e:
            self.logger.error(f"Dashboard generation failed: {e!s}")
            return (
                f"<html><body><h1>ダッシュボード生成エラー: {e!s}</h1></body></html>"
            )

    def _generate_prediction_summary(self, data: VisualizationData) -> str:
        """予測サマリー生成"""
        if not data.predictions:
            return "<p>予測データがありません</p>"

        latest_prediction = data.predictions[-1] if data.predictions else None

        if not latest_prediction:
            return "<p>最新の予測がありません</p>"

        summary_html = f"""
        <div style='padding: 15px;'>
            <h4>最新予測</h4>
            <p><strong>予測価格:</strong> ¥{latest_prediction.get("prediction", 0):.2f}</p>
            <p><strong>信頼度:</strong> {latest_prediction.get("confidence", 0) * 100:.1f}%</p>
            <p><strong>精度:</strong> {latest_prediction.get("accuracy", 0):.1f}%</p>

            <h4>システム情報</h4>
            <p><strong>予測モード:</strong> {latest_prediction.get("mode", "Unknown")}</p>
            <p><strong>処理時間:</strong> {latest_prediction.get("prediction_time", 0) * 1000:.1f}ms</p>

            <h4>パフォーマンス</h4>
        """

        for metric, value in data.performance_metrics.items():
            if isinstance(value, (int, float)):
                summary_html += f"<p><strong>{metric}:</strong> {value:.3f}</p>"

        summary_html += "</div>"
        return summary_html


class DataFetchError(RuntimeError):
    """データ取得エラー"""


class PredictionDashboard:
    """予測可視化ダッシュボード

    特徴:
    - リアルタイム予測可視化
    - インタラクティブチャート
    - センチメント分析表示
    - パフォーマンス監視
    """

    def __init__(
        self,
        enable_web_server: bool = False,
        prediction_service: Optional[Any] = None,
        sentiment_service: Optional[Any] = None,
        max_retries: int = 1,
    ):
        self.logger = logging.getLogger(__name__)
        self.enable_web_server = enable_web_server
        self.dashboard_generator = DashboardGenerator()
        self.prediction_service = prediction_service
        self.sentiment_service = sentiment_service
        self.max_retries = max(0, int(max_retries))

        # Webサーバー設定
        self.app = None
        if enable_web_server and DASH_AVAILABLE:
            self._initialize_web_server()

        self.logger.info("PredictionDashboard initialized")

    def _initialize_web_server(self):
        """Webサーバー初期化"""
        try:
            self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
            self._setup_layout()
            self._setup_callbacks()
            self.logger.info("Web server initialized")
        except Exception as e:
            self.logger.error(f"Web server initialization failed: {e!s}")
            self.app = None

    def _setup_layout(self):
        """レイアウト設定"""
        if not self.app:
            return

        self.app.layout = dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H1(
                                    "ClStock 予測ダッシュボード",
                                    className="text-center mb-4",
                                ),
                                html.Hr(),
                            ],
                        ),
                    ],
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Dropdown(
                                    id="symbol-dropdown",
                                    options=[
                                        {
                                            "label": "ソニーグループ (6758.T)",
                                            "value": "6758.T",
                                        },
                                        {
                                            "label": "トヨタ自動車 (7203.T)",
                                            "value": "7203.T",
                                        },
                                        {
                                            "label": "三菱UFJ (8306.T)",
                                            "value": "8306.T",
                                        },
                                    ],
                                    value="6758.T",
                                ),
                            ],
                            width=4,
                        ),
                        dbc.Col(
                            [dbc.Button("更新", id="refresh-button", color="primary")],
                            width=2,
                        ),
                    ],
                    className="mb-4",
                ),
                dbc.Row(
                    [
                        dbc.Col([dcc.Graph(id="prediction-chart")], width=8),
                        dbc.Col([dcc.Graph(id="sentiment-chart")], width=4),
                    ],
                ),
                dbc.Row(
                    [dbc.Col([html.Div(id="performance-metrics")])], className="mt-4",
                ),
                # 自動更新用
                dcc.Interval(
                    id="interval-component",
                    interval=30 * 1000,  # 30秒ごと
                    n_intervals=0,
                ),
            ],
            fluid=True,
        )

    def _setup_callbacks(self):
        """コールバック設定"""
        if not self.app:
            return

        @self.app.callback(
            [
                Output("prediction-chart", "figure"),
                Output("sentiment-chart", "figure"),
                Output("performance-metrics", "children"),
            ],
            [
                Input("symbol-dropdown", "value"),
                Input("refresh-button", "n_clicks"),
                Input("interval-component", "n_intervals"),
            ],
        )
        def update_dashboard(symbol, n_clicks, n_intervals):
            return self.generate_live_components(symbol)

    def _ensure_visualization_data(
        self, raw_data: Union[VisualizationData, Dict[str, Any]], symbol: str
    ) -> VisualizationData:
        if isinstance(raw_data, VisualizationData):
            data = raw_data
        elif isinstance(raw_data, dict):
            data = VisualizationData(
                symbol=raw_data.get("symbol", symbol),
                predictions=raw_data.get("predictions", []),
                historical_data=self._to_dataframe(raw_data.get("historical_data")),
                sentiment_data=raw_data.get("sentiment_data"),
                performance_metrics=raw_data.get("performance_metrics", {}),
                timestamp=raw_data.get("timestamp", datetime.utcnow()),
            )
        else:
            raise TypeError("Unsupported visualization data type")

        if data.historical_data is None:
            data.historical_data = pd.DataFrame()
        elif not isinstance(data.historical_data, pd.DataFrame):
            data.historical_data = pd.DataFrame(data.historical_data)

        if data.predictions is None:
            data.predictions = []

        if not data.symbol:
            data.symbol = symbol

        return data

    def _to_dataframe(self, value: Any) -> pd.DataFrame:
        if value is None:
            return pd.DataFrame()
        if isinstance(value, pd.DataFrame):
            return value
        return pd.DataFrame(value)

    def _fetch_visualization_data_with_retry(self, symbol: str) -> VisualizationData:
        if not self.prediction_service:
            raise DataFetchError("予測サービスが設定されていません")

        attempts = self.max_retries + 1
        last_error: Optional[Exception] = None

        for attempt in range(attempts):
            try:
                raw_data = self.prediction_service.get_visualization_data(symbol)
                return self._ensure_visualization_data(raw_data, symbol)
            except Exception as exc:
                last_error = exc
                self.logger.warning(
                    "Failed to fetch visualization data for %s (attempt %s/%s): %s",
                    symbol,
                    attempt + 1,
                    attempts,
                    exc,
                )

        message = str(last_error) if last_error else "不明なエラー"
        raise DataFetchError(message)

    def _build_prediction_figure(self, data: VisualizationData) -> go.Figure:
        fig = go.Figure()

        historical = data.historical_data
        if not historical.empty:
            close_series = historical.get("Close")
            if close_series is not None:
                fig.add_trace(
                    go.Scatter(
                        x=list(historical.index),
                        y=list(close_series),
                        mode="lines",
                        name="実際価格",
                    )
                )

        if data.predictions:
            pred_times = []
            pred_values = []
            for pred in data.predictions:
                timestamp = pred.get("timestamp")
                value = pred.get("prediction")
                if timestamp is not None and value is not None:
                    pred_times.append(pd.to_datetime(timestamp))
                    pred_values.append(value)

            if pred_times:
                fig.add_trace(
                    go.Scatter(
                        x=pred_times,
                        y=pred_values,
                        mode="lines+markers",
                        name="予測価格",
                    )
                )

        fig.update_layout(title=f"{data.symbol} 予測チャート", xaxis_title="日時", yaxis_title="価格")
        return fig

    def _build_sentiment_figure(self, sentiment_data: Dict[str, Any]) -> go.Figure:
        if not sentiment_data:
            return self._build_sentiment_error_figure("センチメントデータが利用できません")

        if not PLOTLY_AVAILABLE:
            fig = go.Figure()
            fig.update_layout(title="センチメントデータ")
            fig.add_annotation(
                text="Plotlyが利用できません", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False
            )
            return fig

        figure = go.Figure()
        score = sentiment_data.get("current_sentiment", {}).get("score", 0)
        figure.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=score,
                title={"text": "センチメント"},
                gauge={"axis": {"range": [-1, 1]}},
            )
        )

        return figure

    def _build_sentiment_error_figure(self, message: str) -> go.Figure:
        fig = go.Figure()
        fig.update_layout(title="センチメントデータ")
        fig.add_annotation(
            text=f"センチメント取得に失敗: {message}",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        return fig

    def _build_alert(self, message: str, color: str = "info") -> Any:
        if DASH_AVAILABLE:
            return dbc.Alert(message, color=color)
        return message

    def _build_metrics_component(self, metrics: Dict[str, Any]) -> Any:
        if not metrics:
            return self._build_alert("パフォーマンス指標を取得できませんでした", color="warning")

        items = []
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                text = f"{key}: {value:.3f}"
            else:
                text = f"{key}: {value}"
            items.append(html.Li(text))

        return html.Div([html.H4("パフォーマンス指標"), html.Ul(items)])

    def generate_live_components(self, symbol: str) -> Tuple[go.Figure, go.Figure, Any]:
        try:
            visualization_data = self._fetch_visualization_data_with_retry(symbol)
        except DataFetchError as exc:
            message = f"データ取得に失敗: {exc}"
            prediction_fig = self._build_error_figure(f"{symbol} 予測チャート", message)
            sentiment_fig = self._build_sentiment_error_figure("データが利用できません")
            metrics = self._build_alert(message, color="danger")
            return prediction_fig, sentiment_fig, metrics

        prediction_fig = self._build_prediction_figure(visualization_data)

        sentiment_data = visualization_data.sentiment_data
        sentiment_error: Optional[str] = None
        if self.sentiment_service:
            try:
                fetched_sentiment = self.sentiment_service.get_sentiment(symbol)
                if fetched_sentiment:
                    sentiment_data = fetched_sentiment
            except Exception as exc:
                sentiment_error = str(exc)
                self.logger.warning(
                    "Failed to fetch sentiment data for %s: %s", symbol, sentiment_error
                )
        elif not sentiment_data:
            sentiment_error = "サービスが設定されていません"

        if sentiment_error and not sentiment_data:
            sentiment_fig = self._build_sentiment_error_figure(sentiment_error)
        else:
            sentiment_fig = self._build_sentiment_figure(sentiment_data or {})

        metrics_component = self._build_metrics_component(
            visualization_data.performance_metrics or {}
        )

        return prediction_fig, sentiment_fig, metrics_component

    def _build_error_figure(self, title: str, message: str) -> go.Figure:
        fig = go.Figure()
        fig.update_layout(title=title)
        fig.add_annotation(
            text=message,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        return fig

    def create_dashboard(self, visualization_data: VisualizationData) -> str:
        """ダッシュボード作成"""
        return self.dashboard_generator.generate_static_dashboard(visualization_data)

    def save_dashboard(
        self, visualization_data: VisualizationData, output_path: str = "dashboard.html",
    ):
        """ダッシュボード保存"""
        try:
            dashboard_html = self.create_dashboard(visualization_data)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(dashboard_html)

            self.logger.info(f"Dashboard saved to {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Dashboard save failed: {e!s}")
            return False

    def run_web_server(
        self, host: str = "127.0.0.1", port: int = 8050, debug: bool = False,
    ):
        """Webサーバー起動"""
        if not self.app:
            self.logger.error("Web server not initialized")
            return

        try:
            self.logger.info(f"Starting web server at http://{host}:{port}")
            self.app.run_server(host=host, port=port, debug=debug)
        except Exception as e:
            self.logger.error(f"Web server failed to start: {e!s}")

    def get_dashboard_status(self) -> Dict[str, Any]:
        """ダッシュボード状況取得"""
        return {
            "web_server_available": DASH_AVAILABLE,
            "plotly_available": PLOTLY_AVAILABLE,
            "web_server_enabled": self.enable_web_server,
            "app_initialized": self.app is not None,
        }
