#!/usr/bin/env bash
while true;
do
PID=`cat IPFS.pid` # ID of the process that is uploads to IPFS blockchain

if ! ps -p $PID > /dev/null
then
  rm IPFS.pid
  python3 upload.py
  python3 main.py & echo $! >>IPFS.pid
fi
sleep 3; 
done