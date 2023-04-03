#!/bin/bash 
entries=($(shuf -i 0-20 -n 10000000 -r))
echo "${entries[@]}"