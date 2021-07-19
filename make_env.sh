ENVNAME="nvis_venv"
python3.7 -m venv $ENVNAME
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
