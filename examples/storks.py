import csv
from collections import defaultdict

import pooch
import tqdm

# https://datarepository.movebank.org/entities/datapackage/6c481a15-2f87-4418-9476-8bbd84974adf/full
STORK_DATA_LINK = "https://datarepository.movebank.org/server/api/core/bitstreams/ed3529e6-ce20-4237-8875-9d35fdbf9a0f/content"


def main() -> None:
    stork_csv = pooch.retrieve(url=STORK_DATA_LINK, known_hash=None)

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

    print(len(stork_dict.keys()))


if __name__ == "__main__":
    main()
