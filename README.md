# DenseNoiser
## Noise cancellation model
with this project i wanted to better understand voice enhancement models and made my own
## For explination and rewiev in russian see [EXPLINATION.md](https://github.com/Kozak-Alexandr/DenseNoiser/blob/main/EXPLINATION.md)

# Installation

### For Data and pretrained model go to [Yandex Disc](https://disk.yandex.ru/d/dglhBJF_MW0hrQ)

### clone repository or download needed files to start training from ground zero.
This is but a quick guide to start training on all operating systems, including windows. For other OS you can download all new libraries by hand.

Install Anaconda or create a virtual enviroment similarly
```
conda create --name <env_name> python=3.8
pip install -r requirements.txt
```
if on windows or run into tf errors:

```
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

Anything above 2.10 is not supported on the GPU on Windows Native

```
python -m pip install "tensorflow<2.11"
```

Verify the installation:

```
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### To download pretrained model visit [Yandex Disc](https://disk.yandex.ru/d/dglhBJF_MW0hrQ)
