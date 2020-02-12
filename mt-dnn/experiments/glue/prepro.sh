#! /bin/sh
python3 experiments/glue/glue_prepro.py
python3 prepro_std.py --model bert-base-uncased --root_dir data/canonical_data --task_def experiments/glue/glue_task_def.yml --do_lower_case $1
