import time
import zipfile

import numpy as np
import pooch
from frechetlib.continuous_frechet import frechet_c_approx

ZIP_HASH = "ad5a1d8585769e36bda87a018573831d145af75e8303a063f3de64ed35ed1da9"
ZIP_URL = "http://sarielhp.org/misc/blog/24/05/10/data_e.zip"


def main() -> None:
    curve_data_zip = pooch.retrieve(
        url=ZIP_URL,
        known_hash=ZIP_HASH,
    )

    data_archive = zipfile.ZipFile(curve_data_zip, "r")
    factor = 4.0

    p_curve = np.genfromtxt(data_archive.open("data/test/001_p.plt"), delimiter=",")
    q_curve = np.genfromtxt(data_archive.open("data/test/001_q.plt"), delimiter=",")
    print("Starting workload.")
    ratio, morphing = frechet_c_approx(p_curve, q_curve, factor)
    print("Workload complete")
    print(ratio, morphing.dist)

    print("Timing workload")
    start = time.perf_counter()
    frechet_c_approx(p_curve, q_curve, factor)
    time_taken = time.perf_counter() - start
    print(time_taken)


if __name__ == "__main__":
    main()
