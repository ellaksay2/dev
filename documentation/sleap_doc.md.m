# Sherlock SLEAP (running remote training and inference)

## Pre-Setup

First, start an interactive shell on a compute node with a gpu. _(The prompt will change to show the active job.)_

```
[sunetid@shxx-xxnxx login ~]
$ sdev -g 1 # -g flag is for gpu
[sunetid@shxx-xxnxx login ~]
 (job xxxxxxx) $
```

Notes from Sherlock IT team:
* Before starting, may want to remove any conda/mamba code from your .bashrc ( the lines between "conda initialize" )


**Install SLEAP using Conda**

Note: SLEAP team recommends using Mamba, but Sherlock IT recommended install with Conda as Mamba is deprecated 

To install in home directory:

```
mkdir /home/groups/giocomo/SUNETID
cd /home/groups/giocomo/SUNETID
wgethttps://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
bash Anaconda3-2024.10-1-Linux-x86_64.sh
```

This will start the conda install and we want it to into your $GROUP_HOME

Answer the license questions and pay attention for when it asks where you want to install to.

Put: 
```
/home/groups/giocomo/SUNETID/anaconda3
```

Answer the conda init question with "yes"

Once the install is done, log out of Sherlock and log back in.
Your command prompt should look like:

```
(base)[sunetid@shxx-xxxx login ~]$
```

Now we can start on the install:
```
sdev -g 1 -c 4 -m 8G -t 02:00:00
conda deactivate
ml purge
conda create --name sleap pip python=3.7 cudatoolkit=11.3 cudnn=8.2
```

Note: Sherlock IT suggested 3.7.12 but I got an error saying not accesible by library, SOLVED by downgrading to python=3.7

This will take a little while to resolve dependancies and such but it will ask you to say yes before installing

Once that is installed you can do:

```
conda activate sleap
pip install sleap[pypi]==1.3.3
```

Test the installation was successful by using the following command (you should see the same output)

```
python -c "import sleap; sleap.versions()"
2024-10-28 11:05:11.153284: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /home/groups/ruthm/mthartma/anaconda3/envs/sleap/lib/python3.7/site-packages/cv2/../../lib64:
2024-10-28 11:05:11.153335: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
SLEAP: 1.3.3
TensorFlow: 2.8.4
Numpy: 1.21.6
Python: 3.7.12
```

## Running training/inference in Sherlock

*In SLEAP GUI* 

Predict -> Run training... -> change options to suit your data -> Export training job package... -> Extract compressed .zip file in directory of choice

*In Sherlock* 

Allocate resources for compute node

```
sdev -g 1 -c 4 -m 8G -t 02:00:00
```

Load modules pre-installed on sherlock to allow GPU to work.

```
 (job xxxxxxx) $ ml cuda/11.2.0 # compute on GPU
 (job xxxxxxx) $ ml cudnn/8.1.1.33 # accelerated deep neural network primitives
 (job xxxxxxx) $ ml system qt # several versions of qt are on Sherlock, import one to environemnt
```

Activate conda environment using `conda activate` and navigate to direcotry containing extract training files. 

Run `sleap-train .json .pkg.slp` command 

**Trouble-shooting**
- If killed after a small number of epochs:
    - training config: reduce_on_plateau=FALSE
    - training config: stop_training_on_plateau=FALSE
    - request more resources from Sherlock
    - see [here](https://github.com/talmolab/sleap/discussions/1964)
- 
