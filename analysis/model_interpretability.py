"""モデルの解釈性向上（SHAP等の導入）"""

import logging

import matplotlib.pyplot as plt
import shap

import numpy as np

logger = logging.getLogger(__name__)


class ModelInterpretability:
    """モデルの解釈性向上クラス"""

    def __init__(self, model, X_train, feature_names=None):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.explainer = None

    def setup_explainer(self, method="tree"):
        """SHAP Explainerのセットアップ

        Args:
            method: Explainerの種類 ('tree', 'linear', 'deep', 'kernel')

        """
        if method == "tree":
            self.explainer = shap.TreeExplainer(self.model)
        elif method == "linear":
            self.explainer = shap.LinearExplainer(self.model, self.X_train)
        elif method == "deep":
            self.explainer = shap.DeepExplainer(self.model, self.X_train)
        elif method == "kernel":
            predictor = getattr(self.model, "predict", None)
            if predictor is None:
                predictor = self.model
            self.explainer = shap.KernelExplainer(predictor, self.X_train)
        else:
            raise ValueError(f"サポートされていないmethod: {method}")

        logger.info(f"SHAP Explainerセットアップ完了: {method}")

    def calculate_shap_values(self, X_sample):
        """SHAP値の計算

        Args:
            X_sample: サンプルデータ

        Returns:
            shap_values: SHAP値

        """
        if self.explainer is None:
            raise ValueError(
                "explainerがセットアップされていません。setup_explainer()を呼び出してください。",
            )

        shap_values = self.explainer.shap_values(X_sample)
        return shap_values

    def plot_summary(self, shap_values, X_sample, plot_type="bar", show=True):
        """SHAP要約プロット

        Args:
            shap_values: SHAP値
            X_sample: サンプルデータ
            plot_type: プロットタイプ ('bar', 'dot', 'violin')
            show: プロットを表示するか

        """
        if self.feature_names is not None:
            shap.summary_plot(
                shap_values,
                X_sample,
                feature_names=self.feature_names,
                plot_type=plot_type,
                show=show,
            )
        else:
            shap.summary_plot(shap_values, X_sample, plot_type=plot_type, show=show)

    def plot_waterfall(self, shap_values, X_sample, sample_index=0, show=True):
        """SHAPウォーターフォールプロット（1つのサンプルの詳細）

        Args:
            shap_values: SHAP値
            X_sample: サンプルデータ
            sample_index: プロットするサンプルのインデックス
            show: プロットを表示するか

        """
        # 特定のサンプルのSHAP値を選択
        sample_shap_values = (
            shap_values[sample_index] if len(shap_values.shape) > 1 else shap_values
        )
        sample_X = X_sample[sample_index] if len(X_sample.shape) > 1 else X_sample

        shap.waterfall_plot(
            shap.Explanation(
                values=sample_shap_values,
                base_values=self.explainer.expected_value,
                data=sample_X,
                feature_names=self.feature_names,
            ),
            max_display=15,
            show=show,
        )

    def plot_force(self, shap_values, X_sample, sample_index=0, show=True):
        """SHAPフォースプロット（1つのサンプルの理由）

        Args:
            shap_values: SHAP値
            X_sample: サンプルデータ
            sample_index: プロットするサンプルのインデックス
            show: プロットを表示するか

        """
        # 特定のサンプルのSHAP値を選択
        sample_shap_values = (
            shap_values[sample_index] if len(shap_values.shape) > 1 else shap_values
        )
        sample_X = X_sample[sample_index] if len(X_sample.shape) > 1 else X_sample

        shap.force_plot(
            self.explainer.expected_value,
            sample_shap_values,
            sample_X,
            feature_names=self.feature_names,
            matplotlib=True,
            show=show,
        )

    def get_feature_importance(self, shap_values):
        """特徴量の重要度を取得

        Args:
            shap_values: SHAP値

        Returns:
            feature_importance: 特徴量の重要度

        """
        # SHAP値の絶対値の平均を特徴量の重要度とする
        feature_importance = np.mean(np.abs(shap_values), axis=0)
        return feature_importance

    def plot_feature_importance(self, shap_values, X_sample, top_n=20, show=True):
        """特徴量の重要度プロット

        Args:
            shap_values: SHAP値
            X_sample: サンプルデータ
            top_n: 上位何位まで表示するか
            show: プロットを表示するか

        """
        feature_importance = self.get_feature_importance(shap_values)

        # 重要度が高い順に並べる
        sorted_idx = np.argsort(feature_importance)[::-1][:top_n]

        if self.feature_names is not None:
            top_features = [self.feature_names[i] for i in sorted_idx]
        else:
            top_features = [f"Feature_{i}" for i in sorted_idx]

        top_importance = feature_importance[sorted_idx]

        # プロット
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(top_importance)), top_importance)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel("SHAP Importance")
        plt.title(f"Feature Importance (Top {top_n})")
        plt.gca().invert_yaxis()

        if show:
            plt.show()
        else:
            plt.close()


class AdvancedInterpretability:
    """高度な解釈性分析クラス"""

    def __init__(self, model, X_train, feature_names=None):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.interpretability = ModelInterpretability(model, X_train, feature_names)

    def multi_method_analysis(self, X_sample):
        """複数のSHAPメソッドで分析

        Args:
            X_sample: サンプルデータ

        Returns:
            results: 各メソッドの結果

        """
        results = {}

        # Tree Explainer
        try:
            self.interpretability.setup_explainer("tree")
            tree_shap_values = self.interpretability.calculate_shap_values(X_sample)
            results["tree"] = tree_shap_values
        except Exception as e:
            logger.warning(f"Tree Explainerエラー: {e}")

        # Linear Explainer
        try:
            self.interpretability.setup_explainer("linear")
            linear_shap_values = self.interpretability.calculate_shap_values(X_sample)
            results["linear"] = linear_shap_values
        except Exception as e:
            logger.warning(f"Linear Explainerエラー: {e}")

        # Kernel Explainer
        try:
            self.interpretability.setup_explainer("kernel")
            kernel_shap_values = self.interpretability.calculate_shap_values(X_sample)
            results["kernel"] = kernel_shap_values
        except Exception as e:
            logger.warning(f"Kernel Explainerエラー: {e}")

        return results

    def compare_methods(self, X_sample, methods=["tree", "linear", "kernel"]):
        """複数のSHAPメソッドの結果を比較

        Args:
            X_sample: サンプルデータ
            methods: 比較するメソッドのリスト

        """
        shap_values_dict = {}

        for method in methods:
            try:
                self.interpretability.setup_explainer(method)
                shap_values = self.interpretability.calculate_shap_values(X_sample)
                shap_values_dict[method] = shap_values
            except Exception as e:
                logger.warning(f"{method} Explainerエラー: {e}")

        # 各メソッドの結果をプロットして比較
        for method, shap_values in shap_values_dict.items():
            plt.figure(figsize=(10, 6))
            self.interpretability.plot_feature_importance(
                shap_values,
                X_sample,
                show=False,
            )
            plt.title(f"Feature Importance by {method.upper()} Method")
            plt.show()
