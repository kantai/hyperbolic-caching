#!/bin/bash

cd /disk/local/blanks/cost-cache/node-sim


if [[ "$(pidof node)" != "" ]];  
then kill -s SIGINT $(pidof node)
sleep 5
fi

k=$1
node init.js --k=$k > /tmp/node_stdout &
