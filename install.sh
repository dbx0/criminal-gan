#!/usr/bin/env bash
#Installing penv

pip install virtualenv
virtualenv venv_cgan
source venv_cgan/bin/activate
pip install -r requirements.txt

sh util/install_dataset.sh