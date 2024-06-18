ENVNAME="nvis_venv"
if [ -d $ENVNAME ]
then
    echo "Removing previous environment: $ENVNAME"
    set +e
    rsync -aR --remove-source-files $ENVNAME ~/.Trash/ || exit 1
    set -e
    rm -R $ENVNAME
else
    echo "No previous environment - ok to proceed"
fi
python3.11 -m venv $ENVNAME
source $ENVNAME/bin/activate
pip3 install --upgrade pip  # be sure pip is up to date in the new env.
pip3 install wheel  # seems to be missing (note singular)
pip3 install cython
# # if requirements.txt is not present, create:
# # pip install pipreqs
# # pipreqs
#
# #Then:
#
pip3 install -r requirements.txt
source $ENVNAME/bin/activate

# # build the mechanisms
# # this may equire a separate install of the standard NEURON package
# # with the same version as we have provided
# nrnivmodl cnmodel/mechanisms

source $ENVNAME/bin/activate
python setup.py develop
