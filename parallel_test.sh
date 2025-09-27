sleep 2h
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --standalone --nproc_per_node=4 parallel_minimax_remover.py \
    --test_json grounding_multi_instance_gray.json \
    --base_path /scratch3/yan204/yxp/Senorita/ \
    --model_path ./minimax-remover \
    --video_length 81 \
    --iterations 6