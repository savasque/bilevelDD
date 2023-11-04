#!/bin/bash 

#newgrp rop

DATA="/home/fisch/bilevel/test/data"
EXE="/home/fisch/bilevel/src/bilevel" 
VAI="/home/fisch/vai_qsub"                                               
VAIQUI="/home/fisch/qui_qsub"                                               

sep='_'
#for FILE in `cat redlist.txt` ; do   
#for FILE in `grep gs500c-1 list.txt` ; do

FILTER="*.mps"

for TIMELIMIT in 3600; do   
	K=10000
	for TESTBED in tedConverted RAND_BILEVEL miplib3conv ; do   
		for FILE in `(cd ${DATA}/${TESTBED} && ls $FILTER)` ; do
			let K=$K+1   
			for SETTING in 11 12 13 14 15 16 ; do
				NAME=r${sep}$TESTBED${sep}${FILE}${sep}s$SETTING${sep}t$TIMELIMIT${sep}k$K                             
				$VAI "$EXE -f ${DATA}/${TESTBED}/$FILE -razor 1 -time_limit $TIMELIMIT -setting $SETTING" $NAME.log    
			done
		done
	done
done
