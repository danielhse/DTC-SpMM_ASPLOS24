export DTC_HOME=$(pwd)

export GLOG_PATH="${DTC_HOME}/third_party/glog"
export SPUTNIK_PATH="${DTC_HOME}/third_party/sputnik"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$GLOG_PATH/build/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SPUTNIK_PATH/build/sputnik
