from __future__ import annotations

from typing import Any


class AliasChoices:
    def __init__(self, *aliases: str) -> None:
        self.aliases = aliases


class ConfigDict(dict):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)


def Field(
    default: Any = ..., **kwargs: Any
) -> Any:  # noqa: ANN401 - mimic pydantic signature
    return default


class BaseModel:
    def __init__(self, **data: Any) -> None:
        for key, value in data.items():
            setattr(self, key, value)

    def model_dump(self) -> dict[str, Any]:
        return self.__dict__.copy()


__all__ = ["AliasChoices", "BaseModel", "ConfigDict", "Field"]
