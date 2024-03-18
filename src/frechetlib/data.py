#! /usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import typing as t

_T = t.TypeVar("_T")


class Singleton(type, t.Generic[_T]):
    """
    Singleton metaclass, adapted from here:
    https://stackoverflow.com/a/75308084/2923069
    """

    _instances: t.Dict[Singleton[_T], _T] = {}

    def __call__(cls, *args: t.Any, **kwargs: t.Any) -> _T:
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class FrechetDownloader(metaclass=Singleton):
    _instance: t.Optional[FrechetDownloader] = None

    def __init__(self) -> None:
        print("stuff")
