# Transientangelo: Few-Viewpoint Surface Reconstruction Using Single-Photon Lidar 

![teaser](images/teaser.png)

## 🔨 Installation

1. Create a new conda environment (make sure miniconda3 is installed beforehand). We tested on python version 3.8.13
```
conda create -n env_name python=3.8.13
conda activate env_name
```
2. Additionally please install PyTorch from (https://pytorch.org/get-started/previous-versions/). We tested on pytorch1.13.0 with cuda11.6. Be aware of some issues between NerfAcc and other Pytorch versions here: https://github.com/nerfstudio-project/nerfacc/issues/207

```
# torch 1.13.0+cu116
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
```
3. Install tinycudann (https://github.com/NVlabs/tiny-cuda-nn)
```
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

4. Install the requirements file with 

```
pip install -r requirements.txt
```

4. Install torch-scatter using conda
```
conda install pytorch-scatter -c pyg
```

5. Install
## 🖨️ Dataset 




## 👨‍🍳 Usage 


## 🎓 Citation
```
@inproceedings{luo2024transientangelo,
  title = {Transientangelo: Few-Viewpoint Surface Reconstruction Using Single-Photon Lidar}, 
  author = {Luo, Weihan and Malik, Anagh and Lindell, David B.},
  journal = {arXiv},
  year = {2024}
}
```

## 📣 Acknowledgements
We thank [TransientNeRF](https://github.com/anaghmalik/TransientNeRF) for their implementation of transient volume rendering. We thank [NerfAcc](https://www.nerfacc.com) and [instant-nsr-pl](https://github.com/bennyguo/instant-nsr-pl) for their implementation efficient ray marching and surface-based rendering.
