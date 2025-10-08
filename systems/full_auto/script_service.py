from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

from data_retrieval_script_generator import generate_colab_data_retrieval_script


Printer = Callable[[str], None]
ScriptGenerator = Callable[[Sequence[str]], str]


class DataRetrievalScriptService:
    """Service responsible for generating and storing data retrieval scripts."""

    def __init__(
        self,
        *,
        output_dir: Optional[Path | str] = None,
        script_generator: Optional[ScriptGenerator] = None,
        logger: Optional[logging.Logger] = None,
        printer: Optional[Printer] = None,
        file_name: str = "colab_data_fetcher.py",
        encoding: str = "utf-8-sig",
    ) -> None:
        self._output_dir = Path(output_dir) if output_dir is not None else Path("data") / "retrieval_scripts"
        self._script_generator = script_generator or self._default_script_generator
        self._logger = logger or logging.getLogger(self.__class__.__name__)
        self._printer = printer or print
        self._file_name = file_name
        self._encoding = encoding

    def generate(self, failed_symbols: Iterable[str]) -> Optional[Path]:
        symbols = [symbol for symbol in failed_symbols if symbol]
        self._logger.info("Starting data retrieval script generation. failed_symbols: %s", symbols)
        self._printer(
            f"[INFO] DataRetrievalScriptService.generate called. failed_symbols: {symbols}"
        )

        if not symbols:
            self._logger.info("No failed symbols detected; skipping script generation.")
            self._printer(
                "[INFO] No failed symbols detected. Skipping Google Colab script generation."
            )
            return None

        for index, symbol in enumerate(symbols):
            self._logger.debug("Failed symbol #%d: %s", index, symbol)

        try:
            script_contents = self._script_generator(symbols)
        except Exception:
            self._logger.exception("generate_colab_data_retrieval_script failed")
            self._printer("[ERROR] Failed to generate Google Colab data retrieval script.")
            return None

        if not script_contents or not script_contents.strip():
            self._logger.warning("Generated script is empty.")
            self._printer(
                "[WARNING] Generated data retrieval script is empty. Nothing will be written."
            )
            return None

        output_dir = self._prepare_output_dir()
        script_path = output_dir / self._file_name

        try:
            script_path.write_text(script_contents, encoding=self._encoding)
        except UnicodeEncodeError:
            self._logger.exception("UnicodeEncodeError while writing %s", script_path)
            self._printer(
                f"[ERROR] Unicode encoding error while writing {script_path}."
            )
            return None
        except Exception:
            self._logger.exception("Unexpected error while writing %s", script_path)
            self._printer(
                f"[ERROR] Unexpected error while writing {script_path}."
            )
            return None

        self._logger.info("Saved Google Colab data retrieval script to %s", script_path)
        self._printer(
            f"[INFO] Saved Google Colab data retrieval script to {script_path}"
        )
        return script_path

    def _prepare_output_dir(self) -> Path:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._logger.info("Script output directory prepared: %s", self._output_dir)
        self._printer(f"[INFO] Script output directory: {self._output_dir}")
        return self._output_dir

    @staticmethod
    def _default_script_generator(symbols: Sequence[str]) -> str:
        return generate_colab_data_retrieval_script(
            missing_symbols=list(symbols),
            period="1y",
            output_dir=".",
        )
