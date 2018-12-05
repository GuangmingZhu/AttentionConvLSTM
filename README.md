# AttentionConvLSTM
## Prerequisites
1) Python 2.7
2) Tensorflow-1.2 <br/>
3) The implementation files of the variants of ConvLSTM are in the local dir "patchs". You need merge them with the corresponding files of TF-1.2. <br/> <br/>
   
## Get the pretrained models
The trained models can be obtained from the below link:  <br/>
    Link: https://pan.baidu.com/s/1O-U_Q-5i9wxOA0MDyi3Idg Code: immi

## How to use the code
### Prepare the data
1) Convert each video files into images.
2) Replace the path "/ssd/dataset" in the files under "dataset_splits" 
### Training 
1) Use training_*.py to train the networks for different datasets and different modalities. <br/>
### Testing 
1) Use testing_*.py to evaluate the trained networks on the valid or test subsets of Jester or IsoGD. <br/>

## Citation
Please cite the following paper if you feel this repository useful. <br/>
http://ieeexplore.ieee.org/abstract/document/7880648/
http://openaccess.thecvf.com/content_ICCV_2017_workshops/w44/html/Zhang_Learning_Spatiotemporal_Features_ICCV_2017_paper.html
```
@article{ZhuNIPS2018,
  title={Attention in Convolutional LSTM for Gesture Recognition},
  author={Liang Zhang and Guangming Zhu and Lin Mei and Peiyi Shen and Syed Afaq Shah and Mohammed Bennamoun},
  journal={NIPS},
  year={2018}
}
@article{ZhuICCV2017,
  title={Learning Spatiotemporal Features using 3DCNN and Convolutional LSTM for Gesture Recognition},
  author={Liang Zhang and Guangming Zhu and Peiyi Shen and Juan Song and Syed Afaq Shah and Mohammed Bennamoun},
  journal={ICCV},
  year={2017}
}
@article{Zhu2017MultimodalGR,
  title={Multimodal Gesture Recognition Using 3-D Convolution and Convolutional LSTM},
  author={Guangming Zhu and Liang Zhang and Peiyi Shen and Juan Song},
  journal={IEEE Access},
  year={2017},
  volume={5},
  pages={4517-4524}
}
```

## Contact
For any question, please contact
```
  gmzhu@xidian.edu.cn
```

