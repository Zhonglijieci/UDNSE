# UDNSE: A Universal DNN Neuron State Encryption Framework

*KEYWORDS：Neural networks, Cryptography, Deep learning, Adversarial examples, encryption*


## Description

The code in this repository implements the UDNSE algorithm proposed in the paper, and we will explain how to use this code to validate the contribution points in the paper. In addition, the core part of the code has detailed comments explaining the purpose of these codes.


Please note that our algorithm involves random processes, and the data in the paper is the average of multiple counts. It is normal for the output of the program to differ slightly from the results of the paper, but the trend and magnitude of the data should be the same.

## Overview

- Environment Installation (compiler + dependency packages) [30 human-minutes for newcomer or 10 human-minutes for machine Learning Practitioner]
- Fully Automatic Operation of Encryption and Decrytion Process [3 human-minutes + 1~20 computer-minutes(Based on the size of the carrier and learning rate) ]

## Environment Installation

- Our code is implemented in `python`. Firstly, you need to create a virtual environment. We recommend using `conda` to manage your virtual environment

```bash
conda create -n DNSSE python=3.10
```

- Then, please run under the root directory of this repository:

```bash
cd DNSSE
pip install -r requirements.txt
```

## Experiments

- We provide a simple example where you can run the following code to steganograph a single image:

```bash
cd DNSSE
python demo.py
```
(When running for the first time, you need to download pre trained models from Torch, which takes some time.)

This will use our preset carrier image and key image to perform encryption, and save the steganographic result in the 'result' folder.

- To check the decoding of steganographic images, run:

```bash
cd DNSSE
python decryption.py
```

- To specify the carrier image, key image (with a specified size of 224 * 224 pixels), or plain text, run:

```bash
cd DNSSE
python demo.py --image_path "put/your/image/path/here" --key_path "put/your/image/path/here" --text “Put your massage here”
```

To use a larger carrier image to embed more text, modify the parameter't 'according to the image size, for example:` 448*448:  t=2`, `672*672: t=3`, `896*896: t=4`......

```bash
cd DNSSE
python demo.py --image_path "put/your/image/path/here" --key_path "put/your/image/path/here" --text “Put your massage here” --t 4
```

- In addition, our code supports the vast majority of image classification models pre trained in torch, such as `AlexNet`, `ConvNet`, `Efficientent`, `Inception-v3`, etc. To implement encryption on different models, run:

```bash
cd DNSSE
python demo.py --models wide_resnet50
```

## Other configurable parameters

- `save_path`: Save path for steganographic results
- `transparency`: The coefficient when the carrier image is overlaid with the key, default value is ` 0.5`
- `m_len`: The maximum encoding length for plain text, if text exceeding this length will be truncated. Therefore, when using longer plain text, this parameter needs to be increased, default value is `2400`
- `k`: The number of bits encoded in 1-bit code, default value is `2`
- `split_size`: Split image size, default value is `224`
- `lr`: Learning rate, default value is `1e-4`, Increasing this value slows down the encryption speed and enhances concealment. Conversely, decreasing this value increases the encryption speed and reduces concealment
- `momentum`: Momentum of SGD optimizer, default value is `0.9`

## Reproducibility
A note for hardware: All experiments we run use one RTX 4090 GPU.
