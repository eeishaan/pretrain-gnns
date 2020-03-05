# - load the saved weights
# - Prune weights and save the pruned model
# - Load the pruned model and retrain the model with half number of epochs
# - Freeze the trained weights, batchnorm and biases
# - Proceed normally with

prune_ratio="0.5"
split=species

python packnet.py model_gin/masking.pth tmp/masking_$prune_ratio.mask $prune_ratio

python pretrain_masking.py \
    --num_workers 4 \
    --saved_model model_gin/masking.pth \
    --prune_mask tmp/masking_$prune_ratio.mask \
    --model_file tmp/masking_$prune_ratio &> logs/masking_$prune_ratio.txt

python pretrain_supervised.py \
    --input_model_file tmp/masking_$prune_ratio \
    --output_model_file tmp/supervised_masking_$prune_ratio \
    --prune_mask tmp/masking_$prune_ratio.mask &> logs/supervised_masking_$prune_ratio.txt

python finetune.py \
    --model_file tmp/supervised_masking_$prune_ratio.pth \
    --split $split \
    --filename finetuned_masking_$prune_ratio \
    --epochs 50 \
    --eval_train 1 &> logs/finetuned_masking_$prune_ratio.txt