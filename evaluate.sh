#!/bin/bash
python evaluation.py --gold data/dev_gold.sql --pred generated_datasets/generated_data_augment/saved_models/dev_result.txt --etype all --db data/database/ --table data/tables.json