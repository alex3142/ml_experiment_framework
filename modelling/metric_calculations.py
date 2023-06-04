from collections import defaultdict
import pandas as pd


def calculate_metrics():

    return {"metric_1": 2, "metric_2": 1}


def main():

    mets = defaultdict(list)

    for i in range(5):
        [mets[k].append(v) for k, v in calculate_metrics().items()]

    mets_df = pd.DataFrame(mets)

    mets_df = pd.concat([mets_df, mets_df.apply(['mean'])])

if __name__ == "__main__":
    main()
