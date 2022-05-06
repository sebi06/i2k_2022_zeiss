# I2k_2022_zeiss



This repo contains all the material for the I2K 2022

## Talk

[How to create an open ecosystem for data-centric model development at ZEISS](talk/I2K_ZEISS_Open_ML_Ecosystem.pdf)

## Workshop

How to use the new python packages published on [PyPI]

### Notebooks

* **[cztile]** - A set of tiling utilities for array: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zeiss-microscopy/OAD/blob/master/jupyter_notebooks/cztile/cztile_0_0_2.ipynb)

* **[pylibCZIrw]** - Python wrapper for libCZIrw (C++) library to read and write [CZI] image file format: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zeiss-microscopy/OAD/blob/master/jupyter_notebooks/pylibCZIrw/pylibCZIrw_3_0_0.ipynb)

* **[czmodel]** - A conversion tool for TensorFlow or ONNX ANNs to CZANN

  * For segmentation:&nbsp;
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sebi06/i2k_2022_zeiss/blob/master/workshop/colab_notebooks/SingleClassSemanticSegmentation_3_0_0.ipynb)

  * For regression:&nbsp;
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zeiss-microscopy/OAD/blob/master/Machine_Learning/notebooks/czmodel/Regresssion_3_0_0.ipynb)

### Napari plugin

* **[napari-czann-segment]** - experimental (!) [Napari] plugin to segment images using deep-learning segmentation models trained on [APEER] or in Python stored as a *.czann file 


[Napari]: https://github.com/napari/napari
[PyPI]: https://pypi.org/
[pylibCZIrw]: https://pypi.org/project/pylibCZIrw/
[czmodel]: https://pypi.org/project/czmodel/
[cztile]: https://pypi.org/project/cztile/
[APEER]: https://www.apeer.com
[napari-czann-segment]: https://github.com/sebi06/napari_czann_segment
[CZI]: https://www.zeiss.com/microscopy/int/products/microscope-software/zen/czi.html]

