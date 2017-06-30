# Submits the job to cluster
script=$1
data=$2
PyPath="/home/davidkl/Documents/PyPACE:/home/davidkl/Documents/PyPACE/pyREC"

cluster=davidkl@neutron.phys.ntnu.no
folder=/home/davidkl/Documents/PyPACE
#scp ${data} ${cluster}:${folder}/pyREC/
scp ${script} ${cluster}:${folder}/example/
ssh ${cluster} "PYTHONPATH=${PyPath}; export PYTHONPATH; LD_LIBRARY_PATH=/home/davidkl/privateLib/lib; export LD_LIBRARY_PATH;
nice -19 python ${folder}/${script} ${folder}/${data} > output.txt"
