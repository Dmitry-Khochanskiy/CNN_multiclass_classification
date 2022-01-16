# Image Binary Classification with CNN
Image Binary Classification is a small collection of jupyter notebooks and scripts written for learning of PyTorch and CNN. It consists of a training notebook, an inference notebook, an inference script, and a terminal script, which can be run through a terminal. They can be run on both GPU and CPU. It was tested on WIN 10 with Nvidia GeForce GTC 1660 and RTX 3050.

# Prerequisites
Anaconda, CUDA
Env can be loaded from pytorchenv.yml

# Usage
Inference can be used for inference of a single image or a folder.

# Data set for training
[Melanoma binary classification (https://www.kaggle.com/drscarlat/melanoma)]

# To do
- [ ] Refactor ImgLabeler class 
- [ ] Add universal model loader

# Licence
[MIT](https://choosealicense.com/licenses/mit/)

# Acknowledgments
The [Swapnil Bhange(https://medium.com/@swapnilbhange98/image-classification-between-dog-and-cat-using-resnet-in-pytorch-fdd9fdfac20a)] code was used as a basis for the scripts
Resnet architecture is taken from [Niko Gamulin(https://gist.github.com/nikogamulin/7774e0e3988305a78fd73e1c4364aded)]
Accuracy function for multi class https://towardsdatascience.com/pytorch-vision-multiclass-image-classification-531025193aa
