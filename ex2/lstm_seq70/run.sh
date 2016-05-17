#! /bin/sh
rm -rf data
(time python3 -u ../lstm.py param.yaml) 2>&1 | tee log.txt
