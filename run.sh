# train models
python train_sfold_stage2.py

# choose threshold
python train_sfold_stage2.py --mode=choose_threshold2
python train_sfold_stage2.py --mode=choose_threshold3

# test
python create_submission.py

