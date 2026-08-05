#!/bin/bash

echo "Running all experiments for Greek"
cd StructureTensor
./run_experiments.sh
cd ..
cd LogDet
./run_experiments.sh
cd ..
cd GridSearch
./run_experiments.sh
cd ..
cd Covariance
./run_experiments.sh
cd ..
cd AllHeuristics
./run_experiments.sh
cd ..
cd Zero
./run_experiments.sh
cd ..
cd RefineStructureTensorRhos
./run_experiments.sh
cd ..
echo "Done!"
