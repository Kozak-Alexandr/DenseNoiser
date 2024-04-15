# DenseNoiser
Test task for VK R&amp;D internship 
$ conda create --name <env_name> python=3.8
pip install -r requirements.txt 
if on windows or run into tf errors:
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
# Anything above 2.10 is not supported on the GPU on Windows Native
python -m pip install "tensorflow<2.11"
# Verify the installation:
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
