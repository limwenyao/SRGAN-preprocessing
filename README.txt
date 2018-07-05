Presentation slides: https://docs.google.com/presentation/d/1kOK2lGOjjBvcltexvGphPYB9Rpo_300quQTd0RlQ2_4/edit?usp=sharing

To train SRGAN, open SRGANvgg-tf.ipynb and run (set the appropriate downsample factor 4 or 8). 

To train ResNet, open command line. Run python resnet_cl.py -s [scale] -g[or -ng]. -s to specify the scale of image (1 = ground truth, 4 = downsample by 4, 8 = downsample by 8). To use SRGAN, specify -g, if not specify -ng. Note that you need to have pretrained SRGAN4/SRGAN8 model folders before using -g. 

The folders labelled with Resnet_ prefix contain Resnet classifiers for the following:
Ground Truth
Downsample 4x
Downsample 8x
SRGAN 4x
SRGAN 8x

The folders labeled with SRGAN prefix contain the SRGAN models trained with 4x and 8x downsampled images. 

The image dataset can be downloaded at https://drive.google.com/open?id=1JU7OwH-H1vgZtKhBjjk4maNo6WSlSNi6. Place it within the dataset folder.

The SRGAN 4/8 trained Tensorflow models can be downloaded at https://drive.google.com/open?id=1N4KHKm5P3FiDg-SMx36zSc2lKVZLT9VG.

The ResNet trained Keras models can be downloaded at https://app.box.com/s/7pvho8zujt7nv4ta62nxi7b083z8i4uu.
