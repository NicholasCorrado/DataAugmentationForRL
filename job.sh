#!/bin/bash

# for debugging
pid=$1
step=$2 #ranges from 0 to num_jobs-1
cmd=`tr '*' ' ' <<< $3` # replace * with space
echo $cmd $pid $step

# fetch your code from /staging/
CODENAME=DataAugmentationForRL
cp /staging/ncorrado/${CODENAME}.tar.gz .
tar -xzf ${CODENAME}.tar.gz
rm ${CODENAME}.tar.gz

# fetch conda env from stathing and activate it
ENVNAME=da4rl
ENVDIR=${ENVNAME}
cp /staging/ncorrado/${ENVNAME}.tar.gz .
mkdir $ENVDIR
tar -xzf ${ENVNAME}.tar.gz -C $ENVDIR
rm ${ENVNAME}.tar.gz # remove env tarball
source $ENVDIR/bin/activate

# install editable packages (editable packages cannot be packaged in the env tarball)
cd $CODENAME
pip install -e .
pip install -e custom-envs
cd src

# run your script -- $step ensures seeding is consistent across experiment batches
$($cmd --run_id $step --seed $step)

# compress results. This file will be transferred to your submit node upon job completion.
tar czvf results_${pid}.tar.gz results
mv results_${pid}.tar.gz ../..
