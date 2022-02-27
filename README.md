# SpineGAN

Data from VinDr-SpineXR

conda create --name stylegan2 python=3.7
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3

To prepare data from DICOM format run image_main.py which will convert DICOM to png form and preprocess (crop, downsample) the image.

Run scale_png.py script under dataset tools to preprocess images as detailed in the accompanying milestone report.

Place the images you wish to train on in a folder in the same directory as the train.py script under {NAME}.

To train the model and show the discriminator 2,040,000 images during training using adaptive discriminator augmentation:
python train.py --outdir=training-runs-ada --data={NAME} --gpus=1 --kimg=2040 --aug=ada --target=0.6 --augpipe=bgcfnc --metrics=none --snap=5

To train without adaptive discriminator augmentation:
python train.py --outdir=training-runs-ada --data={NAME} --gpus=1 --kimg=2040 --aug=noaug --metrics=none --snap=5

The classifier can be trained by running all cells where data is provided as DATA_FOLDER >> Subfolder 1: normal png imagings, Subfolder 2: abnormal png imaging
studies in a directory above the model training notebook.
