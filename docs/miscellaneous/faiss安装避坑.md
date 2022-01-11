# faiss 安装避坑
## 安装

更新conda `conda update conda` 

先安装mkl `conda install mkl` 

faiss提供gpu和cpu版，根据服务选择
cpu版本 `conda install faiss-cpu -c pytorch` 
gpu版本 – 记得根据自己安装的cuda版本安装对应的faiss版本，不然会出异常。

使用命令：`nvcc -V` 查看 

`conda install faiss-gpu cudatoolkit=8.0 -c pytorch # For CUDA8` 

`conda install faiss-gpu cudatoolkit=9.0 -c pytorch # For CUDA9 `

`conda install faiss-gpu cudatoolkit=10.0 -c pytorch # For CUDA10` 

## 校验是否安装成功 

`python -c “import faiss”` 

#### tensorflow 对应 cuda 和 cudnn 版本

https://www.tensorflow.org/install/source#gpu