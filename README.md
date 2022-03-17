# drift-predict

This project was created to participate in the [2021 DARPA Forecasting Floats in Turbulence (FFT) Challenge](https://custom.cvent.com/2EA9DACA6FD445A9B9591C3F0C2F58F0/files/6372c91a68444c3ca8b24703e1cccd8b.pdf). This was a live competition for predicting the trajectories of drifters in the Atlantic Ocean. Teams were given access to real-time drifter positions for 20 days, then asked to predict the destiny of these drifters 10 days in the future. A key limitation in the trajectory forecasting is the availability of high-quality marine forecasts. 
Since the conclusion of the competitions, we've collated the data we used to enable its use as a benchmark for developing ocean trajectory forecasting models. 

The competition and dataset timeline is as follows: 

Nov. 2, 2021 - data goes live  
Nov. 22, 2021 - drifter forecasts due  
Dec. 2, 2021 - results announced 


We provide a link to our dataset described as follows: 

- drifter trajectories  
|-- train (Nov. 2 - Nov. 21)  
|-- test  (Nov. 22 - Dec. 2)  

- environmental data:   
|-- [WW3](https://thredds.ucar.edu/thredds/ncss/grib/NCEP/WW3/Global/Best/dataset.html) 3-6 hour wind and waves from Nov. 1 - Nov. 28   
|-- [GFS](https://thredds.ucar.edu/thredds/gfsp5) - 14 day wind forecast from Oct 31 - Dec 3   
|-- [RTOFS](https://nomads.ncep.noaa.gov/pub/data/nccf/com/rtofs/prod/) - 8 day hourly current forecast from Nov. 21 - Nov. 30    


## Setup

- clone and install [Opendrift](https://github.com/opendrift/opendrift)   
- [Download and unzip our collated dataset](https://www.cim.mcgill.ca/~mrl/drift-ncrn/fft-data.zip)   
`wget https://www.cim.mcgill.ca/~mrl/drift-ncrn/fft-data.zip`   
`unzip fft-data.zip`


## To run predictions with ww3 (wave) and gfs (wind) environmental datasets

` python simulate.py -g -w  `

# To evaluate a simulation, pass the results directory to evaluate.py

`python evaluate.py $DATA_DIR/results/spots_N20211121-1530_S20211117-1700_E20211130-1700_DS0_TE0_R1G0W0N0/ `

![Example Prediction](https://github.com/johannah/drift-predict/blob/jrh_argo/media/example_drift.gif) 
