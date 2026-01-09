# Installation

Conda environment
```
conda create -n keygnet python=3.10 -y
conda activate keygnet
```

Install appropriate torch version for your cuda version at /usr/local/cuda
The original keygnet used 11.8, however due to incompatibility with my distro, I used cuda 12.4 drivers with torch cuda12.1 packages

```
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

Install other packages
```
conda install -c conda-forge numpy scipy matplotlib opencv scikit-image scikit-learn pyyaml tqdm tensorboard
python -m pip install albumentations shapely plyfile tensorboardx open3d
```

For monitoring unfortunately tensorboard conflicts with open3d as such you need a separate environment"
```
conda create -n tbonly python=3.10 -y
conda activate tbonly
python -m pip install tensorboard
```
# Creating the dataset

First you need to be inside the keygnet virtual environment
```
conda activate keygnet
```

Then you need to have a ply Stanford triangle mesh inside keygnet/object_files directory
```
mkdir ./object_files
mv YOUR_FILE ./object_files/
```

then proceed to create a BOP style dataset by running (check the file for more customizations)
```
python ./bop_dataset_gen.py --ply-file ./object_files/active_interface.ply --output-dir ./data/BOP/issi --num-frames 1000 --angle-range-deg 30
```

you can also validate this data via the validation file
```
python ./bop_validate.py --dataset-folder ./data/BOP/issi/ --split train
```

# Begin training

Simply run the command inside keygnet environment
```
conda activate keygnet
python train.py --batch_size 8 --keypointsNo 12 --data_root "./data/BOP/issi" --ckpt_root "./data/checkpoints" --epochs 100 logs/issi/radii
```
You can check the train.py file for more arguments and customizations

to monitor the file, open a second terminal and:

Then whenever youre training keyGNet you can monitor the training via
```
conda activate tbonly
tensorboard --logdir ./logs --port 6006
```
you can then head to your browser at http://localhost:6006/ to check the training progress
