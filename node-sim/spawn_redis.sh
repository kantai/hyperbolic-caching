#!/bin/bash

if [[ "$(pidof redis-server)" != "" ]];
then kill -s SIGINT $(pidof redis-server)
sleep 5
fi

port=63791

if [[ "$port" == "" ]];
then
port=63791
fi

strategy=$1

if [[ "$strategy" == "" ]];
then
strategy="hyper"
fi

if [[ "$strategy" == "hyper" ]];
then
redis_dir="redis-hyper"
elif [[ "$strategy" == "hyper-cost-class" ]]
then
redis_dir="redis-cost-class"
elif [[ "$strategy" == "hyper-size" ]]
then
redis_dir="redis-size-aware"
elif [[ "$strategy" == "hyper-size-class" ]]
then
redis_dir="redis-size-aware-class"
else
redis_dir="redis-default"
fi

tmpdir=/tmp/redis_dir_$port
mkdir -p $tmpdir
/disk/local/blanks/cost-cache.git/$redis_dir/src/redis-server /disk/local/blanks/cost-cache.git/$redis_dir/redis.conf --port $port --dir $tmpdir > /tmp/redis-stdout &

