# Investigating and Improving Pre-training of GraphNeural Networks
#### Authors: Ishaan Kumar, Philippe Lelièvre, Saber Benchalel

The code of this project is based on the work from   [Hu et al.](https://github.com/snap-stanford/pretrain-gnns/).

All the code for the new experiments is located in the folder [bio](./bio).  The code from the folder chem remains unchanged.


#### Graph-level self-supervised training of 2nd stage
```
python generate_wl_kernel_dataset.py # to generate WL kernel labels
python pretrain_unsupervised_graph_level.py --pretrained_node_lvl_gnn_path PATH_TO_THE_N0DE_LEVEL_PRETRAIN_MODEL --output_filename OUTPUT_FILE_NAME
python pretrain_infograph.py --pretrained_node_lvl_gnn_path PATH_TO_THE_N0DE_LEVEL_PRETRAIN_MODEL
```
#### Evaluate performance of pre-trained network on node-level tasks
```
python task3.py --weights WEIGHTS_PATH --savepath MODEL_OUTPUT_PATH 
python task3.py --weights WEIGHTS_PATH --savepath MODEL_OUTPUT_PATH --prune_mask PATH_TO_PACK_NET_MASK # with packnet
```
#### Implement PackNet and generate numbers for different thresholds
```
python packnet.py model_gin/masking.pth tmp/masking_$prune_ratio.mask $prune_ratio

python pretrain_masking.py \
    --num_workers 4 \
    --saved_model OUTPUT_MODEL_PATH\
    --prune_mask PATH_TO_PACK_NET_MASK\
    --model_file tmp/masking_$prune_ratio &> OUTPUT_LOG_PATH

python pretrain_supervised.py \
    --input_model_file INPUT_MODEL_PATH \
    --output_model_file OUTPUT_MODEL_PATH \
    --prune_mask PATH_TO_PACK_NET_MASK &> OUTPUT_LOG_PATH

python finetune.py \
    --model_file INPUT_MODEL_PATH \
    --split SPLIT_RATIO \
    --filename OUTPUT_MODEL_PATH \
    --epochs NUM_EPOCH \
    --eval_train 1 &> OUTPUT_LOG_PATH
```
## Original Readme from Hu et al. [here](./original_README.md).