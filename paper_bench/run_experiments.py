import time
import zipfile
from itertools import product

import numpy as np
import pandas as pd
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
    # Hardcoded warmups to make the jit compilation happen
    print("Reading dummy data files.")
    p_curve = np.genfromtxt(data_archive.open("data/test/001_p.plt"), delimiter=",")
    q_curve = np.genfromtxt(data_archive.open("data/test/001_q.plt"), delimiter=",")
    print("Starting warmup.")
    ratio, morphing = frechet_c_approx(p_curve, q_curve, 10.0)
    print("Warmup done.")

    curve_numbers = list(range(1, 9))
    factors = [4.0, 1.1, 1.001]

    # TODO add exact computation and raw VE computation workloads.
    results = []
    for factor, curve_num in product(factors, curve_numbers):
        p_curve = np.genfromtxt(
            data_archive.open(f"data/test/00{curve_num}_p.plt"), delimiter=","
        )
        q_curve = np.genfromtxt(
            data_archive.open(f"data/test/00{curve_num}_q.plt"), delimiter=","
        )

        print(f"Starting workload {curve_num} with approx factor {factor}")
        start = time.perf_counter()
        ratio, morphing = frechet_c_approx(p_curve, q_curve, factor)
        time_taken = time.perf_counter() - start
        print("Workload complete in {time_taken:4f} seconds.")

        res_dict = {
            "Curve Number": curve_num,
            "Approx Factor": factor,
            "Time Taken": time_taken,
        }

        results.append(res_dict)

        df = pd.DataFrame(results)
        df.to_csv("benchmark_results.csv")


if __name__ == "__main__":
    main()
