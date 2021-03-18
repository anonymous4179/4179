# BossNAS
This repository provides the evaluation code of our submitted paper: ***BossNAS: Exploring Hybrid CNN-transformers with Block-wisely Self-supervised Neural Architecture Search***.

![BossPiplineNew](figures/BossPiplineNew.png)

 Illustration of the Siamese supernets training with ensemble bootstrapping.



![SearchSpaceT](figures/SearchSpaceT.png)

Illustration of the fabric-like Hybrid CNN-transformer Search Space with flexible down-sampling positions.



## Our Trained Models 

- Our searched models have been trained from scratch and the ranking files of architectures can be found in the release packege:

    BossNet-T0: **https://github.com/anonymous4179/4179/releases/download/v1.0/BossNet-T0-80_8.pth**
    
    BossNet-T1: **https://github.com/anonymous4179/4179/releases/download/v1.0/BossNet-T1-81_9.pth**
    
- Here is a summary of our searched models:

    |       Model       | MAdds | Steptime | Top-1 (%) | Top-5 (%) |
    | :---------------: | :---: | :------: | :-------: | :-------: |
    | BossNet-T0 w/o SE | 3.4B  |  101ms   |   80.5    |   95.0    |
    |    BossNet-T0     | 3.4B  |  115ms   |   80.8    |   95.2    |
    |    BossNet-T0*    | 5.7B  |  147ms   |   81.6    |   95.6    |
    |    BossNet-T1     | 7.9B  |  156ms   |   81.9    |   95.6    |
    |    BossNet-T1*    | 10.5B |  165ms   |   82.2    |   95.7    |
    
    |            BossNet             | Kendalltau | Spearman | Pearsonr |
    | :----------------------------: | ---------- | -------- | -------- |
    | MBConv search space / ImageNet | 0.65       | 0.78     | 0.85     |
    |     NATS-BenchSS / Cifar10     | 0.53       | 0.73     | 0.72     |




## Usage

### 1. Requirements
- Install PyTorch ([pytorch.org](http://pytorch.org/))
	
- Install third-party requirements

    - We adapt the code from [pytorch-image-models](https://github.com/rwightman/pytorch-image-models/) ( `pip install timm==0.1.14` ),  [OpenSelfSup](https://github.com/open-mmlab/OpenSelfSup) and [DeiT](https://github.com/facebookresearch/deit)  ( you can install *openselfsup* and *deit* refer to the github link ) to validate our models.

- Download the ImageNet dataset and move validation images to labeled subfolders

    - To do this, you can use the following script: [https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.shvalprep.sh](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

      

### 2. Evaluate our models

- You can evaluate our BossNet-T models with the following command:

    Please change the `data-path`, `resume` and `input-size` accordingly.

    ```bash
    python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model bossnet_T0 --input-size 224 --batch-size 128 --data-path /PATH/TO/ImageNet --num_workers 8 --eval --resume PATH/TO/BossNet-T0-80_8.pth
    ```

    ```bash
    python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model bossnet_T1 --input-size 224 --batch-size 128 --data-path /PATH/TO/ImageNet --num_workers 8 --eval --resume PATH/TO/BossNet-T1-81_9.pth
    ```

    

![Boss-T](figures/Boss-T.png)

Architecture of our BossNet-T0

    

- You can get the ranking correlations of BossNAS with the following command:

```bash
cd MBConv_ranking_files
python get_model_score_mbconv.py
```

![BossMBCorrelation](figures/BossMBCorrelation.png)

```bash
cd NATS_SS_ranking_files
python get_model_score_nats.py
```

<img src="figures/ranking_nats_cifar10.png" alt="ranking_nats_cifar10" style="max-width:20%;"/>

## TODO

Training and Searching code will be released later.

