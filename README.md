# dpss-exp3-VC-PPG
Voice Conversion Experiments for THUHCSI Course : &lt;Digital Processing of Speech Signals>
<!-- [![](https://img.shields.io/pypi/v/dpss-exp3-VC-PPG)](https://pypi.org/project/dpss-exp3-VC-PPG/)
![](https://img.shields.io/pypi/pyversions/dpss-exp3-VC-PPG) 
![](https://img.shields.io/pypi/l/dpss-exp3-VC-PPG) -->
This repository provides a PyTorch implementation of Voice-Conversion-PPG Experiments for THUHCSI Course : &lt;Speech Signal Processing>.
This project enables timbre conversion utilizing PPG(Phonetic PosteriorGrams) which can be viewed as the speaker independent linguistic representation.


## Dependencies
- Python 3.6
- Numpy
- Scipy
- PyTorch >= v1.2.0
- librosa
- pysptk
- soundfile
- matplotlib
- wavenet_vocoder ```pip install wavenet_vocoder==0.1.1```
  for more information, please refer to https://github.com/r9y9/wavenet_vocoder


## Data prepration
Download [data](https://drive.google.com/file/d/1KnnUUwkFt9st0lqCpqqSIOBuNBnAeYPp/view?usp=sharing) to ```data```.
Download [pre-trained models](https://drive.google.com/file/d/1JF1WNS57wWcbmn1EztJxh09xU739j4_g/view?usp=sharing) to ```assets```. The assets should look like this:


Download the same WaveNet vocoder model as in [AutoVC](https://github.com/auspicious3000/autovc) to ```assets```

## To Run Demo

```bash
# If you want to specify the GPU:
$ export CUDA_VISIBLE_DEVICES = 0    # set as 0 here, you can change as you need
# Run the inference script which already loads the pretrained conversion model:
$ python inference.py
$ ...
```


## To Train

You can use the provided PPGs to train your model.
```bash
# If you want to specify the GPU:
$ export CUDA_VISIBLE_DEVICES = 0    # set as 0 here, you can change as needed
# Run the training script to launch training process:
$ python main.py
$ ...
```

## To Inference

Change the ckpt path pointing to your trained model.
```Run the inference scripts: python inference.py```

```bash
# If you want to specify the GPU:
$ export CUDA_VISIBLE_DEVICES = 0    # set as 0 here, you can change as you need
# Change the input PPG to ???
$ PPG_path = xxx
# Change the ckpt_path in hparams.py pointing to your model ckpt path:
$ ckpt_path = '/home/jie-wang19/VC-PPG/run/models/660000-G.ckpt'    # change as needed
# Run the inference script to convert:
$ python inference.py
$ ...
```


## Assignment requirements

1. This project is a vanilla voice conversion system based on PPG. 
You need to improve the system performance which is mainly evaluated by the objective evaluation metric MCD.
You can adjust the parameter settings in hparams.py or replace with other vocoders.

2. When you encounter problems while finishing your project, search the [issues](https://github.com/thuhcsi/dpss-exp3-VC-PPG/issues) first to see if there are similar problems.
If there are no similar problems, you can create new issues and state you problems clearly.


## Reference
- [PPG for many-to-one vc](https://ieeexplore.ieee.org/abstract/document/7552917)
- [r9y9/wavenet_vocoder](https://github.com/r9y9/wavenet_vocoder)


