#!/bin/bash

variant=$1
cache=$2

if [[ $cache == "" ]];
then
    echo "Must set cache size (parameter 2)!"
    exit -1
fi

if [[ $variant == "" ]];
then
    echo "Must set cache variant (parameter 1 == default || hyper)!"
    exit -1
fi

if [[ $3 == "" ]];
then
    echo "Must set output (parameter 3)!"
    exit -1
fi

if [[ $4 == "" ]];
then
    echo "Must set sampler (parameter 4)!"
    exit -1
fi

ssh blanks@sns45.cs.princeton.edu "/disk/local/blanks/cost-cache/node-sim/spawn_redis.sh $variant"
ssh blanks@sns44.cs.princeton.edu "/disk/local/blanks/cost-cache/node-sim/spawn_node.sh $cache"

sleep 5

cd /disk/local/blanks/cost-cache/simulation
#python websim.py $4 75 66667 sns44.cs.princeton.edu:3590 "$variant" "$cache" >> results/node-tput/$3
python websim.py $4 100 50000 sns44.cs.princeton.edu:3590 "$variant" "$cache" >> results/node-tput/$3
