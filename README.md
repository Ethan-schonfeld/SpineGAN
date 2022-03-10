# SpineGAN

### Train and Generate images with SpineGAN, using a domain classifier supplemented loss

Implementation of the Schonfeld et al paper: Generative Adversarial Network Based Synthetic Learning and a Novel Domain Relevant Loss Term for Spine Radiographs

### Abstract

Problem: There is a lack of big data for the training of deep learning models in medicine, characterized by the time cost of data collection and privacy concerns. Generative adversarial networks (GANs) offer both the potential to generate new data, as well as to use this newly generated data, without inclusion of patient’s real data, for downstream applications. Approach: A series of GANs were trained and applied for a downstream computer vision spine radiograph abnormality classification task. Separate classifiers were trained with either access or no access to the original imaging. Trained GANs included a conditional StyleGAN2 with adaptive discriminator augmentation (StyleGAN2–ADA), a conditional StyleGAN2 with adaptive discriminator augmentation to generate spine radiographs conditional on lesion type (StyleGAN2–ADA–MultiClass), and using a novel clinical loss term for the generator a StyleGAN2 with adaptive discriminator augmentation conditional on abnormality (SpineGAN). Finally, a differential privacy imposed StyleGAN2–ADA conditional on abnormality was trained and an ablation study was performed on its differential privacy impositions. Key Results: We accomplish GAN generation of synthetic spine radiographs without meaningful input for the first time from a literature review. We further demonstrate the success of synthetic learning for the spine domain with a downstream clinical classification task (AUC of 0.830 using synthetic data compared to AUC of 0.886 using the real data). Importantly, the introduction of a new clinical loss term for the generator was found to increase generation recall as well as accelerate model training. Lastly, we demonstrate that, in a limited size medical dataset, differential privacy impositions severely impede GAN training, finding that this is specifically due to the requirement for gradient perturbation with noise.

### System Requirements

https://github.com/NVlabs/stylegan2

```
conda create --name stylegan2 python=3.7
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3
pip install pandas
pip install sklearn
```

### Train Using Custom Dataset

Preprocess DICOM data:

Navigate to data_processing folder:

```
python image_main.py --DICOM=path to folder with DICOM images --labels=path to csv with condition labels of 0 or 1 under column titled “image_id” --save_0=path to folder to save processed images of class 0 --save_1=path to folder to save processed images of class 1
```

Scale pixel intensities:
```
To for both the class 0 and class 1 folders separately. Processing will be in place
python scale_png.py —directory=path to folder with processed images
```

Make a json file with the image name and a class label (0 or 1)

### SpineGAN Training

To use SpineGAN:

Using a text editor, go to domain-loss-stylegan2-ada-pytorch-main/training/loss.py and specify path to a Binary classification model at line 25 under variable name PATH

Navigate to domain-loss-stylegan2-ada-pytorch-main folder:

To train for 8,000,000 images shown to discriminator and save model after every 200,000 images:
```
To disable adaptive discriminator augmentation or adjust other variables: python train.py --help
python train.py --outdir=output_directory --data=dataset_directory --cond=1 --gpus=1 --kimg=8000 --aug=ada --target=0.6 --augpipe=bgcfnc --metrics=none --snap=50
```
For more information: https://github.com/NVlabs/stylegan2

### Differential Policy StyleGAN2–ADA Training 

Using a text editor, go to DL-stylegan2-ada-pytorch-main/training/training_loop.py

If you wish to disable gradient clipping: comment out line 293.

If you wish to adjust addition of gaussian noise to gradients: line 292.

Gaussian noise currently uses a 0.02 scaling factor which can be adjusted: line 292

If you wish to adjust microbatch size (currently=1): line 103

Navigate to DL-stylegan2-ada-pytorch-main folder:
```
python train.py --outdir=output_directory --data=dataset_directory --cond=1 --gpus=1 --kimg=8000 --aug=ada --target=0.6 --augpipe=bgcfnc --metrics=none --snap=50
```
### Generate SpineGAN Images:

Navigate to domain-loss-stylegan2-ada-pytorch-main folder:
```
python3 generate.py --seeds=0-999 (specify how many images desired) --trunc=1 (specify truncation. default of 1 is no truncation) --network=path to network pkl file  --outdir=output path for generated images --class=(0 or 1) class label desired for generation
```
### Validation and Testing:

In development for later release:

Workaround for now:

Use validation script at training loop of ./abnormality_classification/SpineClassifier.py and specify directory and amount of images 	to make it sample all images of validation/testing set. Load in saved model outside of training loop.

### References:

[1] Martin Abadi, Andy Chu, Ian Goodfellow, H. Brendan
McMahan, Ilya Mironov, Kunal Talwar, and Li Zhang. Deep
learning with differential privacy. CCS ’16, page 308–318,
New York, NY, USA, 2016. Association for Computing Machinery. 2

[2] M.A.P. Chamikara, P. Bertok, I. Khalil, D. Liu, and S.
Camtepe. Privacy preserving distributed machine learning with federated learning. Computer Communications,
171:112–125, 2021. 2

[3] Qi Chang, Hui Qu, Yikai Zhang, Mert Sabuncu, Chao Chen,
Tong Zhang, and Dimitris N. Metaxas. Synthetic learning: Learn from distributed asynchronized discriminator gan
without sharing medical image data. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), June 2020. 2

[4] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li,
and Li Fei-Fei. Imagenet: A large-scale hierarchical image
database. In 2009 IEEE Conference on Computer Vision and
Pattern Recognition, pages 248–255, 2009. 1

[5] Pedro Domingos. A few useful things to know about machine learning. Commun. ACM, 55(10):78–87, oct 2012. 1

[6] Maayan Frid-Adar, Idit Diamant, Eyal Klang, Michal Amitai, Jacob Goldberger, and Hayit Greenspan. Gan-based synthetic medical image augmentation for increased cnn performance in liver lesion classification. Neurocomputing,
321:321–331, 2018. 2

[7] Jeremy Irvin, Pranav Rajpurkar, Michael Ko, Yifan Yu, Silviana Ciurea-Ilcus, Chris Chute, Henrik Marklund, Behzad
Haghgoo, Robyn Ball, Katie Shpanskaya, Jayne Seekins,
David A. Mong, Safwan S. Halabi, Jesse K. Sandberg, Ricky
Jones, David B. Larson, Curtis P. Langlotz, Bhavik N. Patel,
Matthew P. Lungren, and Andrew Y. Ng. Chexpert: A large
chest radiograph dataset with uncertainty labels and expert
comparison. Proceedings of the AAAI Conference on Artificial Intelligence, 33(01):590–597, Jul. 2019. 1

[8] Tero Karras, Miika Aittala, Janne Hellsten, Samuli Laine,
Jaakko Lehtinen, and Timo Aila. Training generative adversarial networks with limited data. In H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and H. Lin, editors, Advances
in Neural Information Processing Systems, volume 33, pages
12104–12114. Curran Associates, Inc., 2020. 3

[9] K Kim, H Cho, T Jang, J Choi, and J Seo. Automatic detection and segmentation of lumbar vertebrae from x-ray images for compression fracture evaluation. Comput Methods
Programs Biomed, Mar 2021. 1

[10] Chan-Pang Kuok, Min-Jun Fu, Chii-Jen Lin, Ming-Huwi
Horng, and Yung-Nien Sun. Vertebrae segmentation from xray images using convolutional neural network. New York,
NY, USA, 2018. Association for Computing Machinery. 1

[11] Maximilian Lam, Gu-Yeon Wei, David Brooks, Vijay Janapa
Reddi, and Michael Mitzenmacher. Gradient disaggregation:
Breaking privacy in federated learning by reconstructing the
user participant matrix. In Marina Meila and Tong Zhang,
editors, Proceedings of the 38th International Conference
on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 5959–5968. PMLR, 18–24
Jul 2021. 2

[12] L Ma, R Shuai, and X Ran. Combining dc-gan with resnet
for blood cell image classification. Med Biol Eng Comput,
58:1251–1264, 2020. 2

[13] Hieu T. Nguyen, Hieu H. Pham, Nghia T. Nguyen, Ha Q.
Nguyen, Thang Q. Huynh, Minh Dao, and Van Vu. Vindrspinexr: A deep learning framework for spinal lesions detection and classification from radiographs. In Marleen de
Bruijne, Philippe C. Cattin, Stephane Cotin, Nicolas Padoy, ´
Stefanie Speidel, Yefeng Zheng, and Caroline Essert, editors,
Medical Image Computing and Computer Assisted Intervention – MICCAI 2021, pages 291–301, Cham, 2021. Springer
International Publishing. 1, 2, 4

[14] H Pham, H Trung, and H Nguyen. Vindr-spinexr: A large
annotated medical image dataset for spinal lesions detection
and classification from radiographs (version 1.0.0). PhysioNet, 2021. 2

[15] TensorFlow. Implement differential privacy with tensorflow
privacy. 2022. 3

[16] TensorFlow. Tensorflow federated: Machine learning on decentralized data. 2022. 3

[17] Praneeth Vepakomma, Otkrist Gupta, Tristan Swedish, and
Ramesh Raskar. Split learning for health: Distributed
deep learning without sharing raw patient data. CoRR,
abs/1812.00564, 2018. 2

[18] Zhuoning Yuan, Yan Yan, Milan Sonka, and Tianbao Yang.
Robust deep AUC maximization: A new surrogate loss and
empirical studies on medical image classification. CoRR,
abs/2012.03173, 2020.
