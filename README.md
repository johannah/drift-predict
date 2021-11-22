# drift-predict




## Setup

- clone and install our [Opendrift](https://github.com/johannah/opendrift-predict.git)
- install cdo (`brew install cdo` on osx)
- set DATA_DIR in utils.py 

## Download data
- download gfs, nces data if desired (not required) 

## To run predictions with rtofs datasets

` python simulate.py -r --download `

# To evaluate a simulation, pass the results directory to evaluate.py
` python evaluate.py /Volumes/seahorse/2021-drifters/results/spots_N20211121-1530_S20211117-1700_E20211130-1700_DS0_TE0_R1G0W0N0/ `

![Example Prediction](https://github.com/johannah/drift-predict/blob/jrh_argo/media/example_drift.gif) 
