set -x
set -e

#nvidia-cuda-mps-control -d


###### CUDA
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# export CUDA_DEVICE_MAX_CONNECTIONS=8
# export CUDA_REDUCE_API_SERIALIZATION=true

###### TF original
# export TF_SYNC_ON_FINISH=false
# export TF_GPU_THREAD_MODE=gpu_private
# export TF_GPU_THREAD_COUNT=1
# export TF_GPU_ALLOCATOR=cuda_malloc_async
# export TF_DETERMINISTIC_ALLOCATOR=true
# export TF_DETERMINISTIC_OPS=true
export TF_FORCE_GPU_ALLOW_GROWTH=true

###### TF multistream
export TF_GPU_CONTEXT_COUNT=5
export TF_GPU_STREAM_GROUP_COUNT=5
export TF_GPU_STREAM_MERGE=true
export TF_PER_STREAM_HOST_ALLOCATOR=true
export TF_GPU_STREAM_GROUP_SHARE_MEMORY_LIMIT=true
export TF_SEGMENT_OWN_CONST=true
export TF_OFFLOAD_ALLOCATOR_MIGRATE=false
# export TF_EXPLICIT_MEMORY_POOL=true
# export TF_ASYNCALLOCATOR_REUSE1=false
# export TF_ASYNCALLOCATOR_REUSE2=false
# export TF_ASYNCALLOCATOR_REUSE3=false

###### TF multistream training
export TF_NODE_LEVEL_MULTISTREAM=true
export TF_REDUCE_STREAM_WAIT=true
#export TF_STREAM_FROM_FILE=/home/robinz/MSTF/tensorflow/tensorflow/cc/tutorials/ple.txt

###### CUDA and TF log
# export TF_MULTI_STREAM_KEY_INFO_LOG=true
# export TF_CPP_MAX_VLOG_LEVEL=5
# export TF_CPP_VMODULE=bfc_allocator=2
# export TF_ENABLE_NVTX_RANGES=1
# export TF_ENABLE_NVTX_RANGES_DETAILED=1
# export TF_DUMP_GRAPH_PREFIX=/home/robinz/coredump/
# export CUDNN_LOGINFO_DBG=1
# export CUDNN_LOGWARN_DBG=1
# export CUDNN_LOGERR_DBG=1
# export CUDNN_LOGDEST_DBG=tf.log
# export CUBLASLT_LOG_LEVEL=2

NSYS=/opt/nvidia/nsight-systems/2023.4.1/bin/nsys

# ${NSYS} profile -o report-ple-masync --force-overwrite true --sample none --cpuctxsw=none --cuda-flush-interval 100 --trace=cuda,osrt,nvtx \
python -u train_tf2_api_offload.py --tf --steps 1000 --no_eval --use_offload #> tmp.log 2>&1


#echo quit | nvidia-cuda-mps-control