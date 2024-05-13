import time
import zipfile

import numpy as np
import pandas as pd
import pooch
from frechetlib.continuous_frechet import frechet_c_approx, frechet_c_compute
from frechetlib.retractable_frechet import retractable_ve_frechet

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
    # frechet_c_approx(p_curve, q_curve, 10.0)
    frechet_c_compute(p_curve, q_curve)
    print("Warmup done.")

    curve_numbers = list(range(1, 9))
    # TODO see what's going on with the lower factor (1.001). This isn't
    # getting stuck in Sariel's code, so something weird may be happening.
    factors = [1.009, 1.1, 4.0]

    # TODO add exact computation and raw VE computation workloads.
    results = []
    for curve_num in curve_numbers:
        try:
            p_curve = np.genfromtxt(
                data_archive.open(f"data/test/00{curve_num}_p.plt"), delimiter=","
            )
            q_curve = np.genfromtxt(
                data_archive.open(f"data/test/00{curve_num}_q.plt"), delimiter=","
            )
        except Exception:
            print(f"Could not read curve file {curve_num}, skipping.")
            continue

        # Run with approx factors first
        for factor in factors:
            print(f"Starting workload {curve_num} with approx factor {factor}")
            start = time.perf_counter()
            ratio, morphing = frechet_c_approx(p_curve, q_curve, factor)
            time_taken = time.perf_counter() - start
            print(f"Workload complete in {time_taken:4f} seconds.")

            res_dict = {
                "Curve Number": curve_num,
                "Algorithm": f"Approx with factor {factor}",
                "Time Taken": time_taken,
                "Distance": morphing.dist,
            }

            results.append(res_dict)

        # Then, run exact
        print(f"Starting workload {curve_num} exact")
        start = time.perf_counter()
        morphing = frechet_c_compute(p_curve, q_curve)
        time_taken = time.perf_counter() - start
        print(f"Workload complete in {time_taken:4f} seconds.")

        res_dict = {
            "Curve Number": curve_num,
            "Algorithm": "Exact",
            "Time Taken": time_taken,
            "Distance": morphing.dist,
        }

        # Finally, run the really slow ve-r
        print(f"Starting workload {curve_num} ve-r")
        start = time.perf_counter()
        morphing = retractable_ve_frechet(p_curve, q_curve, None, None, False)
        time_taken = time.perf_counter() - start
        print(f"Workload complete in {time_taken:4f} seconds.")

        res_dict = {
            "Curve Number": curve_num,
            "Algorithm": "VER",
            "Time Taken": time_taken,
            "Distance": morphing.dist,
        }

        results.append(res_dict)

    # Export test data to CSV file.
    df = pd.DataFrame(results)
    df.to_csv("benchmark_results.csv")


if __name__ == "__main__":
    main()
