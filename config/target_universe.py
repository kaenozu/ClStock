"""Authoritative target universe definitions.

The canonical representation for tickers within the project is the base
Japanese stock code (e.g. ``"7203"``). Exchange-specific variants such
as ``"7203.T"`` are generated on demand via :class:`TargetUniverse`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class TargetSymbol:
    """Metadata for a single target stock symbol."""

    code: str
    english_name: str
    japanese_name: str
    is_core: bool = False


@dataclass(frozen=True)
class TargetUniverse:
    """Container exposing canonical stock codes and formatting helpers."""

    symbols: Tuple[TargetSymbol, ...]
    default_suffix: str = ".T"

    def __post_init__(self) -> None:  # pragma: no cover - simple dataclass wiring
        object.__setattr__(self, "_by_code", {symbol.code: symbol for symbol in self.symbols})

    @property
    def base_codes(self) -> List[str]:
        """Return all canonical stock codes in their base form."""
        return [symbol.code for symbol in self.symbols]

    @property
    def default_codes(self) -> List[str]:
        """Return the subset of core codes used for default trading contexts."""
        defaults = [symbol.code for symbol in self.symbols if symbol.is_core]
        return defaults or self.base_codes

    def format_codes(self, codes: Iterable[str], suffix: str | None = None) -> List[str]:
        """Format the given codes with the desired suffix (``.T`` by default)."""
        suffix = self.default_suffix if suffix is None else suffix
        return [f"{self.to_base(code)}{suffix}" for code in codes]

    def default_formatted(self, suffix: str | None = None) -> List[str]:
        """Return the formatted list of default/core symbols."""
        return self.format_codes(self.default_codes, suffix=suffix)

    def all_formatted(self, suffix: str | None = None) -> List[str]:
        """Return the formatted list of all symbols in the universe."""
        return self.format_codes(self.base_codes, suffix=suffix)

    def variants_for(
        self, code: str, *, suffixes: Sequence[str] | None = None
    ) -> List[str]:
        """Return the base code and formatted variants for the requested symbol."""
        base = self.to_base(code)
        suffixes = tuple(suffixes) if suffixes is not None else (self.default_suffix,)
        variants = [base]
        variants.extend(f"{base}{suffix}" for suffix in suffixes)
        return variants

    @staticmethod
    def to_base(code: str) -> str:
        """Normalise any ticker variant back to its canonical base code."""
        return code.split(".", 1)[0]

    @property
    def english_names(self) -> Dict[str, str]:
        """Mapping of base codes to English company names."""
        return {code: symbol.english_name for code, symbol in self._by_code.items()}

    @property
    def japanese_names(self) -> Dict[str, str]:
        """Mapping of base codes to Japanese company names."""
        return {code: symbol.japanese_name for code, symbol in self._by_code.items()}


# Ordered catalogue of target symbols. Core/default symbols appear first to
# preserve historical behaviour for backtests and demo trading.
_TARGET_SYMBOLS: Tuple[TargetSymbol, ...] = (
    TargetSymbol("6758", "Sony Group Corp", "ソニーグループ", is_core=True),
    TargetSymbol("7203", "Toyota Motor Corp", "トヨタ自動車", is_core=True),
    TargetSymbol("8306", "MUFG Bank", "三菱UFJフィナンシャル・グループ", is_core=True),
    TargetSymbol("9984", "SoftBank Group Corp", "ソフトバンクグループ", is_core=True),
    TargetSymbol("6861", "Keyence Corp", "キーエンス", is_core=True),
    TargetSymbol("4502", "Takeda Pharmaceutical", "武田薬品工業", is_core=True),
    TargetSymbol("6503", "Mitsubishi Electric", "三菱電機", is_core=True),
    TargetSymbol("7201", "Nissan Motor Co", "日産自動車", is_core=True),
    TargetSymbol("8001", "Itochu Corp", "伊藤忠商事", is_core=True),
    TargetSymbol("9022", "East Japan Railway Co", "東日本旅客鉄道", is_core=True),
    TargetSymbol("7267", "Honda Motor Co", "ホンダ"),
    TargetSymbol("7261", "Mazda Motor Corp", "マツダ"),
    TargetSymbol("7269", "Suzuki Motor Corp", "スズキ"),
    TargetSymbol("6902", "Denso Corp", "デンソー"),
    TargetSymbol("6752", "Panasonic Holdings", "パナソニック"),
    TargetSymbol("6701", "NEC Corp", "NEC"),
    TargetSymbol("6702", "Fujitsu Ltd", "富士通"),
    TargetSymbol("6501", "Hitachi Ltd", "日立製作所"),
    TargetSymbol("6502", "Toshiba Corp", "東芝"),
    TargetSymbol("6954", "Fanuc Corp", "ファナック"),
    TargetSymbol("6981", "Murata Manufacturing Co", "村田製作所"),
    TargetSymbol("6971", "Kyocera Corp", "京セラ"),
    TargetSymbol("7751", "Canon Inc", "キヤノン"),
    TargetSymbol("8035", "Tokyo Electron", "東京エレクトロン"),
    TargetSymbol("6770", "Alps Alpine", "アルプスアルパイン"),
    TargetSymbol("9432", "Nippon Telegraph and Telephone", "日本電信電話"),
    TargetSymbol("9433", "KDDI Corp", "KDDI"),
    TargetSymbol("9434", "SoftBank Corp", "ソフトバンク"),
    TargetSymbol("9437", "NTT Docomo Inc", "NTTドコモ"),
    TargetSymbol("6098", "Recruit Holdings Co", "リクルートホールディングス"),
    TargetSymbol("9613", "NTT Data Corp", "NTTデータ"),
    TargetSymbol("8316", "Sumitomo Mitsui Financial Group", "三井住友フィナンシャルグループ"),
    TargetSymbol("8411", "Mizuho Financial Group", "みずほフィナンシャルグループ"),
    TargetSymbol("8604", "Nomura Holdings", "野村ホールディングス"),
    TargetSymbol("8058", "Mitsubishi Corp", "三菱商事"),
    TargetSymbol("8002", "Marubeni Corp", "丸紅"),
    TargetSymbol("8031", "Mitsui & Co", "三井物産"),
    TargetSymbol("8053", "Sumitomo Corp", "住友商事"),
    TargetSymbol("4005", "Sumitomo Chemical Co", "住友化学"),
    TargetSymbol("4063", "Shin-Etsu Chemical Co", "信越化学工業"),
    TargetSymbol("4503", "Astellas Pharma Inc", "アステラス製薬"),
    TargetSymbol("4507", "Shionogi & Co", "塩野義製薬"),
    TargetSymbol("4523", "Eisai Co", "エーザイ"),
    TargetSymbol("4578", "Otsuka Holdings", "大塚ホールディングス"),
    TargetSymbol("5401", "Nippon Steel Corp", "日本製鉄"),
    TargetSymbol("5713", "Sumitomo Metal Mining Co", "住友金属鉱山"),
    TargetSymbol("5020", "ENEOS Holdings", "ENEOS"),
    TargetSymbol("7974", "Nintendo Co", "任天堂"),
    TargetSymbol("8267", "Aeon Co", "イオン"),
    TargetSymbol("9983", "Fast Retailing Co", "ファーストリテイリング"),
    TargetSymbol("3382", "Seven & i Holdings Co", "セブン&アイ・ホールディングス"),
    TargetSymbol("2914", "Japan Tobacco Inc", "日本たばこ産業"),
    TargetSymbol("2802", "Ajinomoto Co", "味の素"),
    TargetSymbol("4911", "Shiseido Co", "資生堂"),
    TargetSymbol("8802", "Mitsubishi Estate Co", "三菱地所"),
    TargetSymbol("8801", "Mitsui Fudosan Co", "三井不動産"),
    TargetSymbol("1801", "Taisei Corp", "大成建設"),
    TargetSymbol("6367", "Daikin Industries Ltd", "ダイキン工業"),
    TargetSymbol("4901", "Fujifilm Holdings", "富士フイルム"),
)

_TARGET_UNIVERSE = TargetUniverse(symbols=_TARGET_SYMBOLS)


def get_target_universe() -> TargetUniverse:
    """Return the immutable global target universe definition."""

    return _TARGET_UNIVERSE


__all__ = ["TargetSymbol", "TargetUniverse", "get_target_universe"]
