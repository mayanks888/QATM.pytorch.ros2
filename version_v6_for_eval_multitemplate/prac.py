from collections import namedtuple
import os
import pandas as pd

csv_path='/media/mayank_sati/DATA/datasets/one_shot_datasets/Farmington/images/2019-09-27-14-39-41_test_scaled_eval_farm.csv'
data = pd.read_csv(csv_path)
print(data.head())
def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

grouped = split(data, 'filename')
for group in sorted(grouped):

        filename = group.filename.encode('utf8')
        obj_ids=list(group.object["obj_id"])
