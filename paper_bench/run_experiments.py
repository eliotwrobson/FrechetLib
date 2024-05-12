import zipfile

import numpy as np
import pooch

ZIP_HASH = "ad5a1d8585769e36bda87a018573831d145af75e8303a063f3de64ed35ed1da9"
ZIP_URL = "http://sarielhp.org/misc/blog/24/05/10/data_e.zip"


def main() -> None:
    curve_data_zip = pooch.retrieve(
        url=ZIP_URL,
        known_hash=ZIP_HASH,
    )

    data_archive = zipfile.ZipFile(curve_data_zip, "r")

    p_curve = np.genfromtxt(data_archive.open("data/test/001_p.plt"), delimiter=",")
    q_curve = np.genfromtxt(data_archive.open("data/test/001_q.plt"), delimiter=",")
    print(p_curve, q_curve)


if __name__ == "__main__":
    main()
