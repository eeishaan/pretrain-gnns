# - load the saved weights
# - Prune weights and save the pruned masking model
# - Load the pruned model, train the pruned weights on contextpred task, saved the 2 task model
# - Load the model and train supervised model, save it
# - Finetune the supervised model on downstream task

prune_ratio="0.5"
split=species

python packnet.py model_gin/masking.pth tmp/masking_$prune_ratio.mask $prune_ratio

python pretrain_masking.py \
    --num_workers 4 \
    --saved_model model_gin/masking.pth \
    --prune_mask tmp/masking_$prune_ratio.mask \
    --model_file tmp/masking_$prune_ratio &> logs/masking_$prune_ratio.txt

python pretrain_contextpred.py \
    --num_workers 4 \
    --saved_model tmp/masking_$prune_ratio.pth \
    --prune_mask tmp/masking_$prune_ratio.mask \
    --model_file tmp/masking_contextpred_$prune_ratio &> logs/masking_contextpred_$prune_ratio.txt

python pretrain_supervised.py \
    --input_model_file tmp/masking_contextpred_$prune_ratio \
    --output_model_file tmp/supervised_masking_contextpred_$prune_ratio &> logs/supervised_masking_contextpred_$prune_ratio.txt

python finetune.py \
    --model_file tmp/supervised_masking_contextpred_$prune_ratio.pth \
    --split $split \
    --filename finetuned_masking_contextpred_$prune_ratio \
    --epochs 50 \
    --eval_train 1 &> logs/finetuned_masking_contextpred_$prune_ratio.txt