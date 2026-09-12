DEVICE=0


###### PART I. Profiling ##########################
###################################################

######### For-loop MHA #################
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=$DEVICE python bench_attention.py --limit-steps 30 --attn looped --prof-dir logs/looped_attn_impl --batch-size 10000 --no-tqdm


######### Vectorized  MHA ###############
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=$DEVICE python bench_attention.py --limit-steps 30 --attn batched --prof-dir logs/batched_attn_impl --batch-size 10000 --no-tqdm

######### SDPA MHA ######################

CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=$DEVICE python bench_attention.py --limit-steps 30 --attn sdpa --prof-dir logs/sdpa_attn_impl --batch-size 10000 --no-tqdm


######### Vectorized  MHA  + faster dataset ###############
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=$DEVICE python bench_attention.py --limit-steps 30 --attn batched --prof-dir logs/batched_attn_impl_fater_dataset --batch-size 10000 --use-faster-dataset --no-tqdm



######### For-loop MHA + compile #################
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=$DEVICE python bench_attention.py --limit-steps 30 --attn looped --prof-dir logs/looped_attn_impl_compile --batch-size 10000 --compile --no-tqdm


######### Vectorized  MHA  + compile###############
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=$DEVICE python bench_attention.py --limit-steps 30 --attn batched --prof-dir logs/batched_attn_impl_compile --batch-size 10000 --compile --no-tqdm

######### SDPA MHA  + compile ######################

CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=$DEVICE python bench_attention.py --limit-steps 30 --attn sdpa --prof-dir logs/sdpa_attn_impl_compile --batch-size 10000 --compile --no-tqdm




###### PART II. torch.utils.bottleneck ############
###################################################
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m torch.utils.bottleneck bench_attention.py --no-profiler --no-tqdm --attn sdpa --limit-steps 50


























############# Change dataset logic ################



CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=1 python bench_attention.py --limit-steps 50 --attn sdpa --prof-dir logs/sdpa_faster_dataset --use-faster-dataset


























###### Part III. torch.compile ####################
###################################################


CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=1 python bench_attention.py --limit-steps 50 --attn sdpa --prof-dir logs/sdpa_attn_impl_compile --compile

CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=1 python bench_attention.py --limit-steps 50 --attn looped --prof-dir logs/looped_attn_impl_compile --compile
