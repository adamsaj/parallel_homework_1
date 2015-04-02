#! /bin/bash

cd autotune

for n in {400..800}; do
	mkdir $n
	cd $n
	cp ~/hw1/scratch/Makefile ~/hw1/scratch/job-blocked ~/hw1/scratch/benchmark.c ~/hw1/scratch/dgemm-blocked.c .
	sed "s/_511_/$n/" dgemm-blocked.c >joy
	mv joy dgemm-blocked.c
	pwd >>~/hw1/scratch/submission_list.txt
	make
	qsub job-blocked
	cd ..
done
cd ..
