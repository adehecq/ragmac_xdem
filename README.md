ragmac_xdem
==============================
[![Build Status](https://github.com/adehecq/ragmac_xdem/workflows/Tests/badge.svg)](https://github.com/adehecq/ragmac_xdem/actions)
[![codecov](https://codecov.io/gh/adehecq/ragmac_xdem/branch/main/graph/badge.svg)](https://codecov.io/gh/adehecq/ragmac_xdem)
[![License:MIT](https://img.shields.io/badge/License-MIT-lightgray.svg?style=flt-square)](https://opensource.org/licenses/MIT)
<!-- [![pypi](https://img.shields.io/pypi/v/ragmac_xdem.svg)](https://pypi.org/project/ragmac_xdem)-->
<!-- [![conda-forge](https://img.shields.io/conda/dn/conda-forge/ragmac_xdem?label=conda-forge)](https://anaconda.org/conda-forge/ragmac_xdem) -->
<!-- [![Documentation Status](https://readthedocs.org/projects/ragmac_xdem/badge/?version=latest)](https://ragmac_xdem.readthedocs.io/en/latest/?badge=latest)-->

Scripts to calculate glacier mass balance as part of the IACS RAGMAC Intercomparison experiment.


## Get started

### Install package
**Note**: To be speed-up the environment setup, you may use [mamba](https://mamba.readthedocs.io). Simply run `conda install mamba -n base -c conda-forge; conda activate base`, then replace all `conda` commands by `mamba`. 
```
git clone git@github.com:adehecq/ragmac_xdem.git ragmac_xdem
cd ragmac_xdem/
conda env create -f environment.yml  # Optional, mostly xdem need to be installed
conda activate ragmac_xdem           # Optional
pip install -e .
```

To update to the latest version of geoutils/xdem (this may not be needed on first install):
```
pip install git+https://github.com/GlacioHack/GeoUtils.git git+https://github.com/GlacioHack/xdem.git --no-deps --force-reinstall
```
or alternatively, you may use your own fork of geoutils/xdem if edits to these codes is needed (see [instructions](https://github.com/GlacioHack/xdem/wiki/Taking-part-to-a-GlacioHack)).

### Download data in ./data/raw folder (~23 GB)
`./scripts/download_data.sh`

### Process data for experiment 2 - WIP
`python ./scripts/main_experiment2.py`

### Process data for experiment 1 - WIP
`python scripts/main_experiment1.py CH_Aletschgletscher`  
`python scripts/main_experiment1.py AT_Hintereisferner`

--------

<p><small>Project based on the <a target="_blank" href="https://github.com/jbusecke/cookiecutter-science-project">cookiecutter science project template</a>.</small></p>
