# standard
import numpy as np

# percentile
from scipy.stats import percentileofscore

def scale_score(x, from_scale_low=-1, from_scale_high=1, to_scale_low=0, to_scale_high=10):

    if from_scale_low < 0:
        temp = from_scale_low

        x -= temp
        from_scale_low -= temp
        from_scale_high -= temp 

    from_scale = (1.0 * x) / (from_scale_high - from_scale_low)
    return (to_scale_high - to_scale_low) * from_scale


def produce_scoring_map(df, col):

    score_dict = {}
    for i in np.linspace(0, 100.01, 10001, endpoint=False):
        score_dict[round(i, 2)] = np.percentile(df[col], round(i, 2))
    return score_dict


def map_to_scoring_map(x, data_column, scoring_map):

    p = round(percentileofscore(data_column, x), 2)
    return scoring_map[p]

