from __future__ import annotations

import typing as t

import numpy as np
import pooch

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
            "01/poly_a.txt": "92f6aebf131aed067c89667dcbd0a22e6d7d9ebb2617a1551ff57703220417e0",
            "01/poly_b.txt": "209ae9dac415df88bcb034f551dc7e85b05e2a269da73f4b3f9966810d4cf6b9",
            "02/poly_a.txt": "baf800aa3dbbd170011d22ad57e4a3f9868013f177a6759a2a1c948ad8c563ea",
            "02/poly_b.txt": "0b498c09e93c2ec30648c979689ca2b859e8310bb82c203c7f9064815f82b118",
            "03/poly_a.txt": "d89c4a1a96673e5a9951e7da915d74b7bb8e9de33771e7384c55a5d1bf4eb310",
            "03/poly_b.txt": "6a3e9c77f34a81901b3dca4362a5b39ceadc504265a1f1f01ebdd7e10b5ba586",
            "04/poly_a.txt": "722caecca8a9917c078aed16797727108eab1ae37102deee08318adc578ac42b",
            "04/poly_b.txt": "93cbbb8d9ed7a3ebb644529d7efbce466f935a296816e8fce11a7d3a2c14e504",
            "05/poly_a.txt": "ccd5641733ac4e7ce686ad12b977671e0324a210ce8bad6a5530b79f3d546b2e",
            "05/poly_b.txt": "5039d1c6acc5f1264891b486ef3cfed9b666dd422aa4f9d483248e9f41c13730",
            "06/poly_a.txt": "3695c2cd531833843c19721b16b372df0c757be590cc51d326cb570d29766d7d",
            "06/poly_b.txt": "e33e13b4c471743925de617ad69e558a61e99ada11f570f88a4cee3e1743332d",
            "07/poly_a.txt": "f1290d3a4aeb57e2d8719f887068272f5307b9bc68593ed3106b9095c8b2f983",
            "07/poly_b.txt": "dfb23dcad24c5637466e4a920502a95323a9bc87573f98c6da93101ec1d84d48",
            "08/poly_a.txt": "3764a36ad68bae12e72a0e59471119f19f06420c460018f008ecfeb4ad75131d",
            "08/poly_b.txt": "db3e195fc24e42da2e01b40439c136057307398a6522978d1ed14120d2bbbce8",
            "09/poly_a.txt": "d0db7e1c00a763c81887466bb1649aa139fe008a77b9a7875f98b70a2cfe97d7",
            "09/poly_b.txt": "ca5a360709f2bd918f527aeefe13e698688b4f031a3b0f4d720ec8b5c5e267ce",
            "10/poly_a.txt": "aef480f1ec3cf7f6953fb4927af7add4c75bc71d80249370db00405fd99f6f71",
            "10/poly_b.txt": "9920187c897e00792b2be50d41712989e66d0b0aba593aba45abad25c8892dc7",
            "11/poly_a.txt": "3ce11a2d31ecccb5e11253bb3a7c38394c1663de6b8ec543fcc7a60d29300209",
            "11/poly_b.txt": "0ac5260a1a2ef94b7469b4fedc7fe6e273626d5c667d5bd1d4abee98b1739f08",
            "12/poly_a.txt": "5c9ae6f14f6789fceadcf6f3b0046a11d6af761746194e27952144102116f22a",
            "12/poly_b.txt": "fafea1bb4e3b9d82fb7fe0402d8cf87e7952458d75b0712dd50193ba273a895e",
            "13/poly_a.txt": "22d765da27feb9bcd0e4834843abb6cf1ace35b889fe85e85cb64893eee84828",
            "13/poly_b.txt": "04aca42b5949ee0d89b17559fde5d063c034814c3b6474b46cfd66b6b1343e6e",
            "14/poly_a.txt": "3695c2cd531833843c19721b16b372df0c757be590cc51d326cb570d29766d7d",
            "14/poly_b.txt": "ef412e395cab0ec0286364602c8d4f8aeee572a7b34825a5939ac0802c8075c4",
            "15/poly_a.txt": "cad0a44d13d6febdb17ac670c256b35ef3770c96fdbdafb95c897a753c962247",
            "15/poly_b.txt": "22ec7ae7ebc2a186058a2733f7677a45d7888dc48cd061fc8b9c5f8e402ab92d",
            "16/poly_a.txt": "2b264a6bf93f215badf25e1c26845d301222fc67a7d1891f9e6856c5567e4399",
            "16/poly_b.txt": "5ba4523c8f2414d46e8a452432e78df41d86568c2f81a38d190de45811b22586",
            "17/poly_a.txt": "a7ca65ce02d3940452276b8c59f2dfbbb8d5f53e05cf0cb7df985f6a432dcc69",
            "17/poly_b.txt": "498a4e6282a0810548727cf896c2223ee56486f744dff1e537ce19b81ba7b3c0",
            "18/poly_a.txt": "734490cfd3d31e23e2583fbe1478ddeefee3be9e9825c45077b24d56acfce877",
            "18/poly_b.txt": "aad18e344670ab9c735a1bf5a3a52b9b64988b7ac57a71fe71ae2f41ec55c490",
            "19/poly_a.txt": "495fd42aceea390bcb9e49785fe1fb0d28bad67423904479d2458d9602d505fd",
            "19/poly_b.txt": "b9b6fd7f0ebe0cd2324c410ba6acbd326e58b1389353ae2f15d87a9b09d579bb",
            "20/poly_a.txt": "495dbe21f03128f0a9576c81e3c4adcbcd7bc3cfe5484a31ab0d6a5653ffb6a3",
            "20/poly_b.txt": "17e8a4748c981e08eb0dff509a8b9235069fef04dc5c858bdc3c67c3a21c09d1",
            "21/poly_a.txt": "f8ee1480e492f63ff94b89a226868ad2d455592361e65756745563eb97746385",
            "21/poly_b.txt": "56f217ad9571a062ee45dd9de903c1b394874ea1f3f07efb4437db2f7f809986",
            "22/poly_a.txt": "495fd42aceea390bcb9e49785fe1fb0d28bad67423904479d2458d9602d505fd",
            "22/poly_b.txt": "82a78fa0c276a5c1135f1a1c5fcf9c6f630afb95eb845d87342908063dc93021",
            "23/poly_a.txt": "dc07a10c20c06e53ce7c101d34ab3a7eeae406326576f9986c9964434f9d6170",
            "23/poly_b.txt": "af4b8e10515ebe27b66e7d9e69bab8890aefcd94541adfe5d22b93664cfb747e",
            "24/poly_a.txt": "f8ee1480e492f63ff94b89a226868ad2d455592361e65756745563eb97746385",
            "24/poly_b.txt": "56f217ad9571a062ee45dd9de903c1b394874ea1f3f07efb4437db2f7f809986",
            "25/poly_a.txt": "b4e0fe013f0fd1a70b941178e1a8bdf58f5d51cdb4af797fd8850ca2791c7bea",
            "25/poly_b.txt": "f145fa23afe9ce251416b2fa3fd72b701b7666461eca0119f10244a2fc4b9c7e",
            "26/poly_a.txt": "e566f71cb0c9b9346f498cce576d09a04ee3066ae39c78cb7b95a6cc08ead73a",
            "26/poly_b.txt": "d162ca79753d121c6a8b9a882fc3bbb7ad3f34feb2935ca8ec115e17a12787f7",
            "27/poly_a.txt": "57d74b3e2f27e5e5f199d1bad65af231744a4028756276128d85599feb7d932a",
            "27/poly_b.txt": "ad07256019966c313302903baf3f48d41cff42ab908a9d8f9073f7c903728e31",
            "28/poly_a.txt": "0c56cace35edb29d31bf28af767b16a6928975f4a3427c70c83c584c753a1cc5",
            "28/poly_b.txt": "60b9bed13f24a0e6857cc60e88d4b34e31bb3760a88e284f4c24e3ff6a048c21",
            "29/poly_a.txt": "acf390a314f476abdfdc45a8aff3b72954935fff4604f941e3c267beb16e13af",
            "29/poly_b.txt": "edfe4729a2e8f7646c3dfda01ca45a796e5aa2e3485b3b04e7fa4a10e877c69d",
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

            num_points = len(lines) - 1

            output_curve: np.ndarray = np.ndarray(
                shape=(num_points, 2), dtype=np.float64
            )

            for i, line in zip(range(num_points), lines):
                curr_point = output_curve[i]
                curr_point[0], curr_point[1] = map(float, line.split(","))

            self.__curves[name] = output_curve

        return self.__curves[name]
