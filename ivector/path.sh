export KALDI_ROOT=`pwd`/kaldi
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C


LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
LD_LIBRARY_PATH=/usr/local/cuda/lib:$LD_LIBRARY_PATH
if [ ! -d /usr/local/cuda/lib64 ]; then
    LD_LIBRARY_PATH=$HOME/usr/local/cuda/lib64:$LD_LIBRARY_PATH
fi

# export CPATH=$HOME/usr/local/cudnn-v5.1/include:/usr/local/cuda/include:$CPATH
# export LIBRARY_PATH=$HOME/usr/local/cudnn-v5.1/lib64:/usr/local/cuda/lib64:/usr/local/cuda/lib:$LIBRARY_PATH

# KERAS_PATH=$(pwd -P)/keras
# HYP_PATH=$(pwd -P)/hyperion

# export MPLBACKEND="agg"


# export PATH=$HYP_PATH/hyperion/bin:/usr/local/cuda/bin:$PATH
# export PYTHONPATH=$HYP_PATH:$KERAS_PATH:$PYTHONPATH
export LD_LIBRARY_PATH
export LC_ALL=C

wait_file() {
    local file="$1"; shift
    local wait_seconds="${2:-30}"; shift # 10 seconds as default timeout
    for((i=0; i<$wait_seconds; i++)); do
	[ -f $file ] && return 1
	sleep 1s
    done
    return 0
}

export -f wait_file
