DEVICE=1


###### PART I. Profiling ##########################
###################################################

######### For-loop MHA #################
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=$DEVICE python bench_attention.py --limit-steps 6 --no-tqdm --attn looped --prof-dir logs/looped_attn_impl


######### Vectorized  MHA ###############
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=$DEVICE python bench_attention.py--limit-steps 6 --no-tqdm --attn batched --prof-dir logs/batched_attn_impl

######### SDPA MHA ######################


CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=1 python bench_attention.py --limit-steps 30 --no-tqdm --attn sdpa --prof-dir logs/sdpa_attn_impl


























###### PART II. torch.utils.bottleneck ############
###################################################
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m torch.utils.bottleneck bench_attention.py --no-profiler --no-tqdm --attn sdpa --limit-steps 50




























############# Change dataset logic ################



CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=1 python bench_attention.py --limit-steps 50 --attn sdpa --prof-dir logs/sdpa_faster_dataset --use-faster-dataset


























###### Part III. torch.compile ####################
###################################################


CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=1 python bench_attention.py --limit-steps 50 --attn sdpa --prof-dir logs/sdpa_attn_impl_compile --compile

CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=1 python bench_attention.py --limit-steps 50 --attn looped --prof-dir logs/looped_attn_impl_compile --compile
