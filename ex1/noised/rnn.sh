#! /bin/sh
rm -rf data
(time python3 -u rnn.py) 2>&1 | tee rnn.log
