#!/bin/bash

cat << "EOF"
   ___ _  _ ___ ___   _    ___  ___   ___  _ _    ___  ___   ___ _  ___   __        ___  ___ ___ 
  / __| \| / __/ __| | |  / _ \/ __| / / \| | |  / _ \/ __| | __| \| \ \ / /  ___  |   \| __| _ \
 | (_ | .` \__ \__ \ | |_| (_) \__ \/ /| .` | |_| (_) \__ \ | _|| .` |\ V /  |___| | |) | _||  _/
  \___|_|\_|___/___/ |____\___/|___/_/ |_|\_|____\___/|___/ |___|_|\_| \_/         |___/|___|_|  
                                                                                                                                                                                                  
EOF

# gnss los/nlos enviornment and dependencies setup
PROJECT_DIR="$(dirname "$0")"
VENV_NAME="venv"
VENV_DIR="$PROJECT_DIR/$VENV_NAME"
REQ_FILE="requirements.txt"

# if the venv does not exists then create
if [ ! -d "$VENV_DIR" ]
then
    echo "Setting up the virtual environment for the project"
    python -m pip install --user virtualenv
    python -m virtualenv $VENV_NAME
    if [ $? -ne 0 ]
    then
        echo "Failed setting up virtual environment for the project"
        exit 1
    fi
    echo "Successfully created virtual environment"
fi

# if the virtual environment is not activated, then activate
if [ -z "$VIRTUAL_ENV" ]
then
    echo "Activating virtual environment"
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]
    then
        source "$VENV_NAME/Scripts/activate"
    fi
else
    echo "$VENV_NAME is already activate"
fi

# installing project dependecies
if [ -f "$REQ_FILE" ]
then
    echo "Installing/Updating the dependencies for the project"
    python -m pip install -r requirements.txt
else
    echo "requirements.txt file is not found"
fi

echo "Setup complete"