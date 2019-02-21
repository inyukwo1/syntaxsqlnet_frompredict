import re
import io
import json
import numpy as np



def merge_files(f1, f2, output_file):
    with open(f1) as inf1:
        data_1 = json.load(inf1)
    with open(f2) as inf2:
        data_2 = json.load(inf2)

    data = data_1 + data_2

    with open(output_file, 'wt') as out:
        json.dump(data, out, sort_keys=True, indent=4, separators=(',', ': '))



merge_files("data/train_all.json", "data/augmented.json", "data/train_all_augmented.json")
