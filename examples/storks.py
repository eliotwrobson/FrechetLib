import csv
import time
from collections import defaultdict

import numpy as np
import pooch
import tqdm
from frechetlib.continuous_frechet import frechet_c_approx

# https://datarepository.movebank.org/entities/datapackage/6c481a15-2f87-4418-9476-8bbd84974adf/full
STORK_DATA_LINK = "https://datarepository.movebank.org/server/api/core/bitstreams/ed3529e6-ce20-4237-8875-9d35fdbf9a0f/content"


def main() -> None:
    stork_csv = pooch.retrieve(
        url=STORK_DATA_LINK,
        known_hash="07df7d22d558f87a7a12bf49d1cf23d13c91596ff028d77c488c79aca667baec",
    )

    stork_dict = defaultdict(list)

    with open(stork_csv) as csvfile:
        stork_reader = csv.reader(csvfile)
        col_labels = next(stork_reader)
        x_idx = col_labels.index("location-long")
        y_idx = col_labels.index("location-lat")
        stork_name_idx = col_labels.index("individual-local-identifier")

        for row in tqdm.tqdm(stork_reader):
            stork_name = row[stork_name_idx]
            x_val = row[x_idx]
            y_val = row[y_idx]

            stork_dict[stork_name].append([x_val, y_val])

    stork_np_dict = dict()

    for stork_name, points in stork_dict.items():
        stork_np_dict[stork_name] = np.array(points, dtype=np.float64)

    # print(stork_np_dict["1787/HH582"].shape, stork_np_dict["1787/HH582"].dtype)
    # exit()
    # print(stork_np_dict["1787/HH582"][100:], stork_np_dict["1791/HE140"][100:])
    print("starting first")
    start_time = time.perf_counter()

    ratio, morphing = frechet_c_approx(
        stork_np_dict["1787/HH582"], stork_np_dict["1791/HE140"], 1.1
    )
    time_taken = time.perf_counter() - start_time

    print("starting", time_taken)

    start_time = time.perf_counter()
    ratio, morphing = frechet_c_approx(
        stork_np_dict["1787/HH582"], stork_np_dict["1791/HE140"], 1.1
    )
    time_taken = time.perf_counter() - start_time
    print(time_taken, ratio, morphing.dist)


if __name__ == "__main__":
    main()
