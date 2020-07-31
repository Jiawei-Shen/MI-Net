# MI-Net

This is the code for our paper "Implicit Euler ODE Networks for Single-Image Dehazing".
[[PAPER]](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w14/Shen_Implicit_Euler_ODE_Networks_for_Single-Image_Dehazing_CVPRW_2020_paper.pdf)

![MI-Net](https://github.com/Jiawei-Shen/MI-Net/blob/master/fig/MI-Net.png)

## Citation

If you find MI-Net useful in your research, please consider citing:

```
@inproceedings{shen2020implicit,
  title={Implicit Euler ODE Networks for Single-Image Dehazing},
  author={Shen, Jiawei and Li, Zhuoyan and Yu, Lei and Xia, Gui-Song and Yang, Wen},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  pages={218--219},
  year={2020}
}
```

## Benchmark Results
![PSNR/SSIM](https://github.com/Jiawei-Shen/MI-Net/blob/master/fig/PSNR_SSIMs.png)

![](https://github.com/Jiawei-Shen/MI-Net/blob/master/fig/OURS.jpg) ![](https://github.com/Jiawei-Shen/MI-Net/blob/master/fig/OUT.jpg)

![](https://github.com/Jiawei-Shen/MI-Net/blob/master/fig/OURSFLOWER.jpg) ![](https://github.com/Jiawei-Shen/MI-Net/blob/master/fig/HAZYFLOWER.jpg)

## Train

```
python main.py
```
Before the training process, you have to reset the parameters in ```main.py```, for an instance, the path of your datasets and result.

For the training dataset, you can use images directly with our dataset building functions in ```create.py``` (Note that the names between input and ground-truth have to be corresponding!) or use the dataset in PyTorch form.

### 
[[RESIDE_Dataset]](https://sites.google.com/view/reside-dehaze-datasets/reside-v0)


## Test

```
python test_dehazy.py
```
Also, you have to reset the path parameters before the image test.

