#! /bin/sh
rm -rf data
(time python3 -u ../gru.py param.yaml) 2>&1 | tee log.txt
