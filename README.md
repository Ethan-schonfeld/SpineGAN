# SpineGAN

Data from VinDr-SpineXR

First pip install stylegan2 (below)
To prepare data from DICOM format run image_main.py which will convert DICOM to png form and preprocess (crop, downsample) the image.
To run model as in the paper use:
stylegan2_pytorch --data training_data/normal --aug-prob 0.25 --name normal-gan --results_dir ./normal_results --models_dir ./normal_models

The classifier can be trained by running all cells where data is provided as DATA_FOLDER >> Subfolder 1: normal png imagings, Subfolder 2: abnormal png imaging
studies in a directory above the model training notebook.

To train the GAN use instructions from: https://github.com/lucidrains/stylegan2-pytorch which this training uses.


To generate, move the checkpoint file you wish to generate from to models/default and name the model: model_NUMBER.pt

stylegan2_pytorch --generate --load-from NUMBER --num-generate NUMBER_OF_IMAGES_TO_GENERATE
