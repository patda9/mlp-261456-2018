#!/bin/bash

EXISTING_PACKAGE=$(dpkg -l | grep python3-venv)

if [ ! -z "$EXISTING_PACKAGE" ];
then
    echo "python3-venv exists"

    EXISTING_VENV=.venv

    if [ ! -d "$EXISTING_VENV" ];
    then
        echo "creating virtual environment under /.venv/"

        python3 -m venv .venv
    fi
else
    echo "Installing python3-venv"
    sudo apt-get install python3-venv
fi

source .venv/bin/activate

pip install -r requirements.txt

echo "Done setting up"
