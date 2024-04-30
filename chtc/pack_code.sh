cd ../.. # cd just outside the repo
tar --exclude="chtc" --exclude='src/results' --exclude='results_chtc' -czvf DataAugmentationForRL DataAugmentationForRL
cp DataAugmentationForRL /staging/ncorrado