#!/bin/bash

# untar python installation and make sure the script uses it
tar -xzf python39.tar.gz
export PATH=$PWD/python/bin:$PATH
rm python39.tar.gz

# fetch your packages from /staging/
cp /staging/ntseng/packages.tar.gz .
tar -xzf packages.tar.gz
rm packages.tar.gz
# make sure python knows where your packages are
export PYTHONPATH=$PWD/packages

# fetch your code from /staging/
CODENAME=DataAugmentationForRL              ########## change when switching repos
cp /staging/ntseng/${CODENAME}.tar.gz .
tar -xzf ${CODENAME}.tar.gz
rm ${CODENAME}.tar.gz

python3 -m pip install --upgrade pip
cd $CODENAME
python3 -m pip install -e .
cd src
python3 -m pip install -e custom-envs

pid=$1
step=$2 #ranges from 0 to num_jobs-1
cmd=`tr '*' ' ' <<< $3` # replace * with space
echo $cmd $pid $step

# run your script
# $step ensures seeding is constistent across experiment batches
$($cmd --run_id $pid --seed $step)

# compress results. This file will be transferred to your submit node upon job completion.
tar czvf results_${pid}.tar.gz results
mv results_${pid}.tar.gz ../..
