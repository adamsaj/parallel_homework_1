#! /bin/bash

for i in `more submission_list.txt`; do echo $i >>register_search.txt; less $i/job-blocked.stdout | grep Average >> register_search.txt; done
