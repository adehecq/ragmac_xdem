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
```
git clone git@github.com:adehecq/ragmac_xdem.git ragmac_xdem
cd ragmac_xdem/
conda env create -f environment.yml  # Optional, mostly xdem need to be installed
conda activate ragmac_xdem           # Optional
pip install -e .
```

### Download data in ./data/raw folder
`./scripts/download_data.sh`

## Process data for experiment 2 - WIP
`python ./scripts/main_experiment2.py`

--------

<p><small>Project based on the <a target="_blank" href="https://github.com/jbusecke/cookiecutter-science-project">cookiecutter science project template</a>.</small></p>
