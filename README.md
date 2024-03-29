# How to cite us
Mimicry embedding for advanced neural network training of 3D biomedical micrographs

Artur Yakimovich, Moona Huttunen, Jerzy Samolej, Barbara Clough, Nagisa Yoshida, Serge Mostowy, Eva Frickel, Jason Mercer

bioRxiv 820076; doi: https://doi.org/10.1101/820076

# ZedMate Readme

ZedMate is a suite of plugins aimed multi-channel intensity quantification.

The main plugin called ZedMate is aimed at particle detection and intensity quantification in 3D multi-channel images. The main detector is based on the TrackMate Laplacian of Gaussian engine. Most contemporary microscopy modalities are anisotropic in their XYZ-resolution. Specifically Z-resolution is inferior to XY. That's why it is convenient to think of particle detection in 3D as a 2D particle detection + tracking problem. ZedMate manuscript is currently in preparation

 ![ZedMate](https://github.com/ayakimovich/ZedMate/blob/master/img/zedmate.png "ZedMate")

# Mimicry embedding

One of the features provide by the ZedMate plugin is the mimicry embedding of the detected particle to resemble known datasets. Benfits of this strategy are discussed in the upcomming publication.
![Mimicry Embedding](https://github.com/ayakimovich/ZedMate/blob/master/img/mimicry_embedding.png "Mimicry Embedding")

# ZedMate Plugin on ImageJ.net
Further description of the plugin is also available on [ImageJ Wiki](https://imagej.net/ZedMate)

# Datasets produced with ZedMate Plugin
We have recently produced an example of the datasets produced using the mimicry embedding feature of ZedMate. Here a single virus particle dataset is embedded to resemble MNIST hand-written digits dataset: [Virus-MNIST](https://github.com/ayakimovich/virus-mnist)

For further questions contact:
[Artur Yakimovich, PhD](mailto:artur.yakimovich@gmail.com)

*Latest update: 22-08-2019*
