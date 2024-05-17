import time
import zipfile

import Fred
import numpy as np
import pandas as pd
import pooch
from frechetlib.continuous_frechet import frechet_c_approx, frechet_c_compute
from frechetlib.retractable_frechet import retractable_ve_frechet

# NOTE change this to True to skip the slow benchmarks
SKIP_MEMORY_INTENSIVE_BENCHMARKS = False


ZIP_HASH = "ad5a1d8585769e36bda87a018573831d145af75e8303a063f3de64ed35ed1da9"
ZIP_URL = "http://sarielhp.org/misc/blog/24/05/10/data_e.zip"
WEIRD_CURVE_NUMS = frozenset((4, 5))


def main() -> None:
    curve_data_zip = pooch.retrieve(
        url=ZIP_URL,
        known_hash=ZIP_HASH,
    )

    data_archive = zipfile.ZipFile(curve_data_zip, "r")
    # Hardcoded warmups to make the jit compilation happen
    print("Reading dummy data files.")
    # TODO add a frechet c compute test with these curves after running with Sariel's code
    p_curve = np.genfromtxt(data_archive.open("data/test/001_p.plt"), delimiter=",")
    q_curve = np.genfromtxt(data_archive.open("data/test/001_q.plt"), delimiter=",")
    print("Starting warmup.")
    frechet_c_compute(p_curve, q_curve)
    print("Warmup done.")

    curve_numbers = list(range(1, 9))
    # TODO see what's going on with the lower factor (1.001). This isn't
    # getting stuck in Sariel's code, so something weird may be happening.
    factors = [4.0, 1.1]  # , 1.01]

    # TODO add exact computation and raw VE computation workloads.
    results = []
    for curve_num in curve_numbers:
        p_curve_file = data_archive.open(f"data/test/00{curve_num}_p.plt")
        q_curve_file = data_archive.open(f"data/test/00{curve_num}_q.plt")

        if curve_num not in WEIRD_CURVE_NUMS:
            p_curve = np.genfromtxt(p_curve_file, delimiter=",")
            q_curve = np.genfromtxt(q_curve_file, delimiter=",")
        else:
            # Parse the geolife curves differently for reasons
            p_curve = np.genfromtxt(
                p_curve_file, delimiter=",", skip_header=6, usecols=(0, 1)
            )
            q_curve = np.genfromtxt(
                q_curve_file, delimiter=",", skip_header=6, usecols=(0, 1)
            )

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

        # Run other algo
        fred_curve_p = Fred.frechet_approximate_minimum_link_simplification(
            Fred.Curve(p_curve), 500
        )
        fred_curve_q = Fred.frechet_approximate_minimum_link_simplification(
            Fred.Curve(q_curve), 500
        )

        print(f"Starting workload {curve_num} fred exact")
        start = time.perf_counter()
        distance = Fred.continuous_frechet(fred_curve_p, fred_curve_q).value
        time_taken = time.perf_counter() - start
        print(f"Workload complete in {time_taken:4f} seconds.")

        res_dict = {
            "Curve Number": curve_num,
            "Algorithm": "Fred Exact",
            "Time Taken": time_taken,
            "Distance": distance,
        }

        results.append(res_dict)

        break

        # Then, run exact
        if not (SKIP_MEMORY_INTENSIVE_BENCHMARKS and curve_num == 4):
            # Skip number 4 for this because my machine runs out of memory lol

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

            results.append(res_dict)

        # Always skip number 4 here because it's way too huge.
        if not (SKIP_MEMORY_INTENSIVE_BENCHMARKS or curve_num == 4):
            # NOTE this doesn't run a lot of the time because I run out of memory
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
