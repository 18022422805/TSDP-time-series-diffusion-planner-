###################################
# User Configuration Section
###################################
NUPLAN_DATA_PATH="D:/Monash/Research/Nuplan/data/cache/mini" # nuplan training data path
NUPLAN_MAP_PATH="D:/Monash/Research/Nuplan/maps" # nuplan map path

TRAIN_SET_PATH="D:/Monash/Research/Nuplan/processed_mini" # preprocess training data
###################################

python data_process.py \
--data_path $NUPLAN_DATA_PATH \
--map_path $NUPLAN_MAP_PATH \
--save_path $TRAIN_SET_PATH \
--total_scenarios 1000000 \

