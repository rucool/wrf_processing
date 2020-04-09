# wrf_processing
Tools for processing and plotting RU-WRF data.


## Installation Instructions
Add the channel conda-forge to your .condarc. You can find out more about conda-forge from their website: https://conda-forge.org/

`conda config --add channels conda-forge`

Clone the wrf_processing repository

`git clone https://github.com/rucool/wrf_processing.git`

Change your current working directory to the location that you downloaded wrf_processing. 

`cd /Users/lgarzio/Documents/repo/wrf_processing/`

Create conda environment from the included environment.yml file:

`conda env create -f environment.yml`

Once the environment is done building, activate the environment:

`conda activate wrf_processing`

Install the toolbox to the conda environment from the root directory of the wrf_processing toolbox:

`pip install .`

The toolbox should now be installed to your conda environment.