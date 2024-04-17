#!/bin/bash

# untar python installation and make sure the script uses it
tar -xzf python39.tar.gz
export PATH=$PWD/python/bin:$PATH
rm python39.tar.gz

# fetch your packages from /staging/
cp /staging/ncorrado/packages.tar.gz .
tar -xzf packages.tar.gz
rm packages.tar.gz
# make sure python knows where your packages are
export PYTHONPATH=$PWD/packages

# fetch your code from /staging/
CODENAME=LessOffPolicy
cp /staging/ncorrado/${CODENAME}.tar.gz .
tar -xzf ${CODENAME}.tar.gz
rm ${CODENAME}.tar.gz

cd $CODENAME
pip install -e .
pip install -e src/custom_envs
pip install gymnasium imageio
cd src

pid=$1
step=$2 #ranges from 0 to num_jobs-1
cmd=$3
echo $cmd $pid $step

# run your script
# $step ensures seeding is constistent across experiment batches
$($cmd --run-id $pid --seed $step)

# compress results. This file will be transferred to your submit node upon job completion.
tar czvf results_${pid}.tar.gz results
