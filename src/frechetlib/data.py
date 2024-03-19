#! /usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import itertools as it
import typing as t

import numpy as np
import pooch

import frechetlib.frechet_utils as fu

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
    """
    Registry for all the example data stored on Sariel's site, with corresponding hashes.
    See: https://sarielhp.org/p/24/frechet_ve/examples/
    """

    __slots__ = ("__curves", "__file_fetcher", "__registry")

    __curves: t.Dict[str, np.ndarray]
    __file_fetcher: pooch.Pooch
    __registry: t.Dict[str, str]

    def __init__(self) -> None:
        self.__registry = {
            "01/poly_a.txt": "f16ed62fd6139dcbc3ef6c474fef9b08aa93f1db17b83f0c4be2a5d352a37399",
            "01/poly_b.txt": "5e0ae148083f75a0bb6c242bc8e874c0827efefef4a46035feae8174fd52e4f6",
            "02/poly_a.txt": "14c199a124c2fd591942ba989a309ef7af137658a745063e65abc8562e3f48a9",
            "02/poly_b.txt": "102b9fc7b5851a91545398fe1b011f7e88f17975a1bc3c70f8bbfaa41083fdff",
            "03/poly_a.txt": "103f96fafc598e2078e8d2ea5b0fca0f88a6564d545c2aca53741ffb83a86edd",
            "03/poly_b.txt": "64c1ba36ee4408b11ac1fa8e16e991165bd4391c9c2453dde6e0030eb487e49f",
            "04/poly_a.txt": "e514c1ac419439fcb903cdf8eb349bbb6129ab3c21548fb0795b13f7e49a15a0",
            "04/poly_b.txt": "a5c8cc42bb6208dc0a15941260bdf5280b4f8afe3637c94d09712a050349bc2e",
            "05/poly_a.txt": "1b461a14d3d15a5a31e5b06e8a89720ec958b3dcc0c846eb7691c159be429ddb",
            "05/poly_b.txt": "798260bfd586cd909cc1c21824531ac3f92ca3848a38c162d12777c144a6bdff",
            "06/poly_a.txt": "104548210f71896058175e9c32367f8c871def79040ee9be5aa49536de102661",
            "06/poly_b.txt": "3a2b4e78ad8d665c12d2c95a6b2a2941e5cf027e7228348ea707b46c33d35f4b",
            "07/poly_a.txt": "3d64a7d4f63f0eabd4754181ef5cc44cbb9c782b670f601604f3e62f04db5f90",
            "07/poly_b.txt": "3b62341f25574b45fcdb639b502e2d868934884d72901fabc4b8d8ae2bc09d7a",
            "08/poly_a.txt": "feab18a3d07ee38261132ffcd6bdc3896d3247912d08021ebceebc802fb7a382",
            "08/poly_b.txt": "ba75d20c98d0208cf34b350ba402d8fa96875d4744b00da9824cac6620c4f077",
            "09/poly_a.txt": "64b56af768c5147c3696c1cb39018f258ae60dc23af9f66a7bedff0bcc06959f",
            "09/poly_b.txt": "9a0a8bcea41cdcec2bdccea8b3cf9ca384b64e9bb40b2014d81dc453fe51e208",
            "10/poly_a.txt": "68efcda07bb68d91ad6c01cb2122e5c0db6b3dc2916690d3ad8e5de102d5584d",
            "10/poly_b.txt": "fa77d5673a8e0c702909e63f54166439c38686e499b61f2302925dcaa58231da",
            "11/poly_a.txt": "a94621bcebc1a654220e42362182eecb5344d7edb98658fb8b06c2f67b9ca081",
            "11/poly_b.txt": "e06e86d0219a74ed156457925b3c410ade8ac5813a4093732a245d6eacd71853",
            "12/poly_a.txt": "c99e208e48275a894b0f28d503f98c9e176e96a6bffbac9ed861974390e7efd3",
            "12/poly_b.txt": "4b11cc56991b697b07957585a6a2ff2c96ce82df401cfeb5b94aba03c7d32349",
            "13/poly_a.txt": "d04590b19e7f8ec58a38a8dca5393fe4e6f7df72a490a6c0fea5e7889d909daa",
            "13/poly_b.txt": "45f4090b8dc0249b8d2f3db187086fb45a7b655cc6abd75ca41330b2a4578f9c",
            "14/poly_a.txt": "104548210f71896058175e9c32367f8c871def79040ee9be5aa49536de102661",
            "14/poly_b.txt": "b97f51dc043812febaabc04f60b25056e8049b20c78500ee190197b39dd0ee48",
            "15/poly_a.txt": "f68beeecbc714407f5563b73d23158e72d5432df45ae2d487f4ff6a43c16caa1",
            "15/poly_b.txt": "e3871347a6369ad2d96ef511efe6599f033ee2d54e75fa29230637b9caeb944f",
        }

        self.__file_fetcher = pooch.create(
            # Use the default cache folder for the operating system
            path=pooch.os_cache("frechetlib"),
            base_url="https://sarielhp.org/p/24/frechet_ve/examples",
            # The registry specifies the files that can be fetched
            registry=self.__registry,
        )

        self.__curves = {}

    def get_curve(self, name: str) -> np.ndarray:
        """
        Get the curved named "name" from this registry.
        """

        if name not in self.__registry:
            raise ValueError(f'File with name "{name}" not found in registry.')

        if name not in self.__curves:
            file_dir = self.__file_fetcher.fetch(name)

            with open(file_dir) as file:
                contents = file.read()

            lines = contents.split("\n")
            x_coords = lines[0].split()
            y_coords = lines[1].split()

            assert len(x_coords) == len(y_coords)

            num_points = len(x_coords)

            output_curve: fu.Curve = np.ndarray(shape=(num_points, 2), dtype=np.float64)

            for i, x_coord, y_coord in zip(it.count(0), x_coords, y_coords):
                output_curve[i][0] = x_coord
                output_curve[i][1] = y_coord

            self.__curves[name] = output_curve

        return self.__curves[name]
