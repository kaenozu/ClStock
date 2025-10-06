import asyncio
import sys
import types
from types import SimpleNamespace

from menu import run_full_auto


def _patch_dependencies(monkeypatch, fake_class):
    """Inject stub dependencies for the full auto system."""
    fake_module = types.ModuleType("full_auto_system")
    fake_module.FullAutoInvestmentSystem = fake_class
    monkeypatch.setitem(sys.modules, "full_auto_system", fake_module)

    def run_coroutine(coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    monkeypatch.setattr(asyncio, "run", run_coroutine)
    monkeypatch.setattr("builtins.input", lambda *args, **kwargs: "")


def test_run_full_auto_displays_recommendations(monkeypatch, capsys):
    recommendation = SimpleNamespace(
        symbol="7203.T",
        company_name="トヨタ自動車",
        recommendation_score=9.5,
        expected_return=5.25,
        risk_level="Low",
        buy_timing="2024-01-01 09:00",
        sell_timing="2024-01-05 15:00",
        reasoning="Strong fundamentals",
    )

    class FakeFullAutoSystem:
        def __init__(self):
            self.recommendations = [recommendation]

        async def run_full_auto_analysis(self):
            return self.recommendations

    _patch_dependencies(monkeypatch, FakeFullAutoSystem)

    run_full_auto()

    captured = capsys.readouterr().out
    assert "フルオート投資推奨結果" in captured
    assert "推奨 1: 7203.T" in captured
    assert "推奨度: 9.5/10" in captured


def test_run_full_auto_handles_empty_recommendations(monkeypatch, capsys):
    class FakeFullAutoSystem:
        async def run_full_auto_analysis(self):
            return []

    _patch_dependencies(monkeypatch, FakeFullAutoSystem)

    run_full_auto()

    captured = capsys.readouterr().out
    assert "現在推奨できる銘柄がありません" in captured


def test_run_full_auto_reports_import_error(monkeypatch, capsys):
    class FakeFullAutoSystem:
        def __init__(self):
            raise ImportError("missing dependency")

    _patch_dependencies(monkeypatch, FakeFullAutoSystem)

    run_full_auto()

    captured = capsys.readouterr().out
    assert "フルオートシステムの読み込みに失敗しました" in captured
    assert "システムが完全にインストールされていない可能性があります" in captured


def test_run_full_auto_reports_runtime_error(monkeypatch, capsys):
    class FakeFullAutoSystem:
        async def run_full_auto_analysis(self):
            raise RuntimeError("analysis failed")

    _patch_dependencies(monkeypatch, FakeFullAutoSystem)

    run_full_auto()

    captured = capsys.readouterr().out
    assert "フルオート実行中にエラーが発生しました" in captured
