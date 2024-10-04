# Mamba-SEUNet: Mamba UNet for Monaural Speech Enhancement (Submitted to ICASSP 2025)

**Abstract:** 
 In recent speech enhancement (SE) research, transformer and its variants have emerged as the predominant methodologies. However, the quadratic complexity of the self-attention mechanism imposes certain limitations on practical deployment. Mamba, as a novel state-space model (SSM), has gained widespread application in natural language processing and computer vision due to its strong capabilities in modeling long sequences and relatively low computational complexity. In this work, we introduce Mamba-SEUNet, an innovative architecture that integrates Mamba with U-Net for SE tasks. By leveraging bidirectional Mamba to model forward and backward dependencies of speech signals at different resolutions, and incorporating skip connections to capture multi-scale information, our approach achieves state-of-the-art (SOTA) performance. Experimental results on the VCTK+DEMAND dataset indicate that Mamba-SEUNet attains a PESQ score of 3.59, while maintaining low computational complexity. When combined with the Perceptual Contrast Stretching technique, Mamba-SEUNet further improves the PESQ score to 3.73.

## Pre-requisites
1. Python >= 3.8.
2. Clone this repository.
3. Install python requirements. Please refer requirements.txt.
4. Download and extract the [VoiceBank+DEMAND dataset](https://datashare.ed.ac.uk/handle/10283/1942).

## Training
For single GPU (Recommend), Mamba-SEUNet needs at least 12GB GPU memery.
```
python train.py
```

## Training with your own data
Generate six dataset json files using data/make_dataset_json.py
```
python make_dataset_json.py
```

## Inference
```
python inference.py --checkpoint_file /PATH/TO/YOUR/CHECK_POINT/g_xxxxxxx
```

## Acknowledgements
We referred to [MP-SENet](https://github.com/yxlu-0102/MP-SENet), [MUSE](https://github.com/huaidanquede/MUSE-Speech-Enhancement), [SEMamba](https://github.com/RoyChao19477/SEMamba)
