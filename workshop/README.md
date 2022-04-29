- [cztile - Python package to simplify the process of tiling arrays](#cztile---python-package-to-simplify-the-process-of-tiling-arrays)
  - [Samples for pylibCZIrw](#samples-for-pylibczirw)
  - [System setup](#system-setup)
  - [Supported Tiling Strategies](#supported-tiling-strategies)
    - [AlmostEqualBorderFixedTotalAreaStrategy2D](#almostequalborderfixedtotalareastrategy2d)
      - [Inputs](#inputs)
      - [Calculation of tile positions and borders](#calculation-of-tile-positions-and-borders)
- [pylibCZIrw - Python wrapper for libCZIrw](#pylibczirw---python-wrapper-for-libczirw)
  - [Important Remarks](#important-remarks)
  - [Example Usage](#example-usage)
  - [Installation](#installation)
- [CZMODEL package](#czmodel-package)
  - [Samples for czmodel](#samples-for-czmodel)
  - [System setup for czmodel](#system-setup-for-czmodel)
  - [Model conversion](#model-conversion)
    - [Keras models in memory](#keras-models-in-memory)
      - [1. Create a model meta data class](#1-create-a-model-meta-data-class)
      - [2 .Creating a model specification](#2-creating-a-model-specification)
      - [3. Converting the model](#3-converting-the-model)
    - [Exported TensorFlow models](#exported-tensorflow-models)
    - [Adding pre- and post-processing layers](#adding-pre--and-post-processing-layers)
    - [Unpacking CZANN/CZSEG files](#unpacking-czannczseg-files)
  - [CZANN Model Specification](#czann-model-specification)
  - [Disclaimer](#disclaimer)

# cztile - Python package to simplify the process of tiling arrays

This project provides simple-to-use tiling functionality for arrays. It is not linked directly to the CZI file format, but can be of use to process such images in an efficient and **tile-wise** manner, which is especially important when dealing with larger images.

## Samples for pylibCZIrw

The basic usage can be inferred from this sample notebook:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zeiss-microscopy/OAD/blob/master/jupyter_notebooks/cztile/cztile_0_0_2.ipynb)

## System setup

The current version of this toolbox only requires a fresh Python 3.x installation.
Being a pure Python wheel, it was tested with Python 3.9 on Windows.

## Supported Tiling Strategies

This package features the following tiling strategies:

### AlmostEqualBorderFixedTotalAreaStrategy2D

This covers a total area with a minimal number of tiles of constant total area such that:

- the image area is completely covered by tiles and is filled up with as few tiles as possible
- the overlap/border between tiles is as small as possible, but it is ensured that at least a minimum border size is used
- all interior tiles have the same size
- a possible excess border is split among tiles and can lead to slightly different tile and border sizes at the edge
- all interior tiles have at least a minimum border width/height on all sides
- the edge tiles have zero border at the edge and at least the minimum border width on their inner sides.
- The sizes of all non-zero borders differ at most by one pixel.

![cztile - AlmostEqualBorderFixedTotalAreaStrategy2D](https://raw.githubusercontent.com/zeiss-microscopy/OAD/master/jupyter_notebooks/cztile/cztile_algo.png)

The core functionality is of course also available for 1D.

The **AlmostEqualBorderFixedTotalAreaStrategy2D** is based on the following algorithm:

#### Inputs

Image width: ![equation](https://latex.codecogs.com/svg.image?W)
Minimum interior border width (left or right): ![equation](https://latex.codecogs.com/svg.image?%5Cdelta)
Fixed total tile width: ![equation](https://latex.codecogs.com/svg.image?w)

#### Calculation of tile positions and borders

**Case 1:** ![equation](https://latex.codecogs.com/svg.image?W%3Cw)
There is no solution. Fail!

**Case 2:** ![equation](https://latex.codecogs.com/svg.image?W=w)
Use a single tile with no borders.

**Case 3:** ![equation](https://latex.codecogs.com/svg.image?W%3Ew)
Maximum inner tile width of edge tiles: ![equation](https://latex.codecogs.com/svg.image?%5Chat%7B%5Comega%7D=w-%5Cdelta)
Maximum inner tile width of interior tiles: ![equation](https://latex.codecogs.com/svg.image?%5Chat%7Bw%7D=w-2%5Cdelta)
Total interior tile width: ![equation](https://latex.codecogs.com/svg.image?%5COmega=%5Cmax%5C%7B%5C0,W-2%5C,%5Chat%7B%5Comega%7D%5C%7D)
Number of tiles: ![equation](https://latex.codecogs.com/svg.image?N=%5Cleft%5Clceil%7B%5COmega/%5Chat%7Bw%7D%7D%5Cright%5Crceil&plus;2)
Excess border: ![equation](https://latex.codecogs.com/svg.image?E=2%5Chat%7B%5Comega%7D&plus;(N-2)%5Chat%7Bw%7D-W)
Total number of non-zero left and right borders: ![equation](https://latex.codecogs.com/svg.image?%5Cnu=2(N-1))
Fractional excess border: ![equation](https://latex.codecogs.com/svg.image?e=E/%5Cnu)
The first non-zero border has index ![equation](https://latex.codecogs.com/svg.image?j=1), the last has index ![equation](https://latex.codecogs.com/svg.image?j=%5Cnu). Tile ![equation](https://latex.codecogs.com/svg.image?i) is surrounded by the borders with index ![equation](https://latex.codecogs.com/svg.image?2i) and ![equation](https://latex.codecogs.com/svg.image?2i&plus;1).
Cumulative excess border for all borders up to border ![equation](https://latex.codecogs.com/svg.image?j): ![equation](https://latex.codecogs.com/svg.image?E_j=%5Cleft%5Clfloor%7Bje%7D%5Cright%5Crfloor) for ![equation](https://latex.codecogs.com/svg.image?j=0,...,%5Cnu)
Cumulative border for all borders up to border ![equation](https://latex.codecogs.com/svg.image?j): ![equation](https://latex.codecogs.com/svg.image?%5CDelta_j=E_j&plus;j%5Cdelta) for ![equation](https://latex.codecogs.com/svg.image?j=0,...,%5Cnu)
Tile boundaries: ![equation](https://latex.codecogs.com/svg.image?x_i=%5Cbegin%7Bcases%7D0%7Ci=0%5C%5Ci%5C,w-%5CDelta_%7B2i-1%7D%7Ci=1,...,N-1%5C%5CW%7Ci=N%5Cend%7Bcases%7D)
Tile ![equation](https://latex.codecogs.com/svg.image?i) for ![equation](https://latex.codecogs.com/svg.image?i=0,...,N-1):

- Left-most pixel of inner tile: ![equation](https://latex.codecogs.com/svg.image?L_i=x_i)
- Right-most pixel of inner tile: ![equation](https://latex.codecogs.com/svg.image?R_i=x_%7Bi&plus;1%7D-1)
- Inner tile width: ![equation](https://latex.codecogs.com/svg.image?w_i=x_%7Bi&plus;1%7D-x_i)
- Total border width: ![equation](https://latex.codecogs.com/svg.image?b_i=w-w_i)
- Left border width: ![equation](https://latex.codecogs.com/svg.image?%5Clambda_i=%5Cbegin%7Bcases%7D0%7Ci=0%5C%5C%5CDelta_%7B2i%7D-%5CDelta_%7B2i-1%7D%7Ci=1,...,N-2%5C%5Cb_i%7Ci=N-1%5Cend%7Bcases%7D)
- Right border width: ![equation](https://latex.codecogs.com/svg.image?%5Crho_i=b_i-%5Clambda_i)
- Left-most border pixel: ![equation](https://latex.codecogs.com/svg.image?%5Cl_i=L_i-%5Clambda_i)
- Right-most-border pixel: ![equation](https://latex.codecogs.com/svg.image?r_i=R_i&plus;%5Crho_i)

------

# pylibCZIrw - Python wrapper for libCZIrw

This project provides a simple and easy-to-use Python wrapper for libCZIrw - a cross-platform C++ library intended for providing read and write access to CZI image documents.

## Important Remarks

- At the moment, **pylibCZIrw** completely abstracts away the subblock concept, both in the reading and in the writing APIs.
- If pylibCZIrw is extended in the future to support subblock-based access (e.g. accessing acquisition tiles), this API must not be altered.
- The core concept of pylibCZIrw is focussing on reading and writing 2D image planes by specifying the dimension indices and its location in order to only read or write **what is really needed**.

## Example Usage

The basic usage can be inferred from this sample notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zeiss-microscopy/OAD/blob/master/jupyter_notebooks/pylibCZIrw/pylibCZIrw_3_0_0.ipynb)

For more detailed information refer to the pylibCZIrw-documentation.html shipped with the source distribution of this package (see the **Download files** section).

## Installation

In case there is no wheel available for your system configuration, you can:

- try to install from the provided source distribution
  **For Windows**:
  - try to [keep paths short on systems with maximum path lengths](https://github.com/pypa/pip/issues/3055)
  - make [Win10 accept file paths over 260 characters](https://www.howtogeek.com/266621/how-to-make-windows-10-accept-file-paths-over-260-characters/)
- reach out to the maintainers of this project to add more wheels

# CZMODEL package

This project provides simple-to-use conversion tools to generate a CZANN file from a
[TensorFlow](https://www.tensorflow.org/) or [ONNX](https://onnx.ai/) model that resides in memory or on disk to be usable in the
[ZEN Intellesis](https://www.zeiss.com/microscopy/int/products/microscope-software/zen-intellesis-image-segmentation-by-deep-learning.html) module starting with ZEN blue >=3.2 and ZEN Core >3.0.

Please check the following compatibility matrix for ZEN Blue/Core and the respective version (self.version) of the CZANN Model Specification JSON Meta data file (see _CZANN Model Specification_ below). Version compatibility is defined via the [Semantic Versioning Specification (SemVer)](https://semver.org/lang/de/).

| Model (legacy)/JSON        | ZEN Blue           | ZEN Core  |
| -------------------------- |:------------------:| ---------:|
| 1.0.0                      | \> 3.4             | \> 3.3    |
| 3.1.0 (legacy)             | \> 3.3             | \> 3.2    |
| 3.0.0 (legacy)             | \> 3.1             | \> 3.0    |

If you encounter a version mismatch when importing a model into ZEN, please check for the correct version of this package.

## Samples for czmodel

For segmentation:&nbsp;
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sebi06/i2k_2022_zeiss/blob/master/workshop/colab_notebooks/SingleClassSemanticSegmentation_3_0_0.ipynb)

For regression:&nbsp;
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zeiss-microscopy/OAD/blob/master/Machine_Learning/notebooks/czmodel/Regresssion_3_0_0.ipynb)

## System setup for czmodel

The current version of this toolbox only requires a fresh Python 3.x installation.
It was tested with Python 3.7 on Windows.

## Model conversion

The toolbox provides a `convert` module that features all supported conversion strategies.
It currently supports converting Keras models in memory or stored on disk with a corresponding metadata JSON file (see _CZANN Model Specification_ below).

### Keras models in memory

The toolbox also provides functionality that can be imported e.g. in the training script used to fit a Keras model.
It provides different converters to target specific versions of the export format. Currently, there are two converters available:

- DefaultConverter: Exports a .czann file complying with the specification below.
- LegacyConverter (Only for segmentation): Exports a .czseg file (version 3.1.0 of the legacy ANN-based segmentation models in ZEN).

The converters are accessible by running:

```python
from czmodel.convert import DefaultConverter, LegacyConverter
```

Every converter provides a `convert_from_model_spec` function that uses a model specification object to convert a model to the corresponding export format. It accepts a `tensorflow.keras.Model` that will be exported to [ONNX](https://onnx.ai/) and in case of failure to [SavedModel](https://www.tensorflow.org/guide/saved_model)
format and at the same time wrapped into a .czann/.czseg file that can be imported and used by Intellesis.
To provide the meta data, the toolbox provides a ModelSpec class that must be filled with the model, a ModelMetadata instance containing the information required by the specification (see _Model Metadata_ below), and optionally a license file.

A CZANN/CZSEG can be created from a Keras model with the following three steps.

#### 1. Create a model meta data class

To export a CZANN, meta information is needed that must be provided through a `ModelMetadata` instance.

For segmentation:

```python
from czmodel.model_metadata import ModelMetadata, ModelType

model_metadata = ModelMetadata(
    input_shape=[1024, 1024, 3],
    output_shape=[1024, 1024, 5],
    model_type=ModelType.SINGLE_CLASS_SEMANTIC_SEGMENTATION,
    classes=["class1", "class2"],
    model_name="ModelName",
    min_overlap=[90, 90]
)
```

For regression:

```python
from czmodel.model_metadata import ModelMetadata, ModelType

model_metadata = ModelMetadata(
    input_shape=[1024, 1024, 3],
    output_shape=[1024, 1024, 3],
    model_type=ModelType.REGRESSION,
    model_name="ModelName",
    min_overlap=[90, 90]
)
```

For legacy CZSEG models the legacy `ModelMetadata` must be used:

```python
from czmodel.legacy_model_metadata import ModelMetadata as LegacyModelMetadata

model_metadata_legacy = LegacyModelMetadata(
    name="Simple_Nuclei_SegmentationModel_Legacy",
    classes=["class1", "class2"],
    pixel_types="Bgr24",
    color_handling="ConvertToMonochrome",
    border_size=90,
)
```

#### 2 .Creating a model specification

The model and its corresponding metadata are now wrapped into a ModelSpec object.

```python
from czmodel.model_metadata import ModelSpec

model_spec = ModelSpec(
    model=model,
    model_metadata=model_metadata,
    license_file="C:\\some\\path\\to\\a\\LICENSE.txt"
)
```

The corresponding model spec for legacy models is instantiated analogously.

```python
from czmodel.legacy_model_metadata import ModelSpec as LegacyModelSpec

legacy_model_spec = LegacyModelSpec(
    model=model,
    model_metadata=model_metadata_legacy,
    license_file="C:\\some\\path\\to\\a\\LICENSE.txt"
)
```

#### 3. Converting the model

The actual model conversion is finally performed with the ModelSpec object and the output path and name of the CZANN.

```python
from czmodel.convert import DefaultConverter

DefaultConverter().convert_from_model_spec(model_spec=model_spec, output_path='some/path', output_name='some_file_name')
```

For legacy models the interface is similar.

```python
from czmodel.convert import LegacyConverter

LegacyConverter().convert_from_model_spec(model_spec=legacy_model_spec, output_path='some/path', output_name='some_file_name')
```

### Exported TensorFlow models

Not all TensorFlow models can be converted. You can convert a model exported from TensorFlow if the model and the
provided meta data comply with the _CZANN Model Specification_ below.

The actual conversion is triggered by either calling:

```python
from czmodel.convert import DefaultConverter

DefaultConverter().convert_from_json_spec('Path to JSON file', 'Output path', 'Model Name')
```

or by using the command line interface of the `savedmodel2czann` script:

```console
savedmodel2ann path/to/model_spec.json output/path/ output_name --license_file path/to/license_file.txt
```

### Adding pre- and post-processing layers

Both, `convert_from_json_spec` and `convert_from_model_spec` in the converter classes accept the
following optional parameters:

- `spatial_dims`: Set new spatial dimensions for the new input node of the model. This parameter is expected to contain the new height
and width in that order. **Note:** The spatial input dimensions can only be changed in ANN architectures that are invariant to the
spatial dimensions of the input, e.g. FCNs.
- `preprocessing`: One or more pre-processing layers that will be prepended to the deployed model. A pre-processing layer must be derived from the `tensorflow.keras.layers.Layer` class.
- `postprocessing`: One or more post-processing layers that will be appended to the deployed model. A post-processing layer must be derived from the `tensorflow.keras.layers.Layer` class.

While ANN models are often trained on images in RGB(A) space, the ZEN infrastructure requires models inside a CZANN to
expect inputs in BGR(A) color space. This toolbox offers pre-processing layers to convert the color space before
passing the input to the model to be actually deployed. The following code shows how to add a BGR to RGB conversion layer
to a model and set its spatial input dimensions to 512x512.

```python
from czmodel.util.transforms import TransposeChannels

# Define dimensions and pre-processing
spatial_dims = 512, 512  # Optional: Target spatial dimensions of the model
preprocessing = [TransposeChannels(order=(2, 1, 0))]  # Optional: Pre-Processing layers to be prepended to the model. Can be a single layer, a list of layers or None.
postprocessing = None  # Optional: Post-Processing layers to be appended to the model. Can be a single layer, a list of layers or None.

# Perform conversion
DefaultConverter().convert_from_model_spec(
    model_spec=model_spec,
    output_path='some/path',
    output_name='some_file_name',
    spatial_dims=spatial_dims,
    preprocessing=preprocessing,
    postprocessing=postprocessing
)
```

Additionally, the toolbox offers a `SigmoidToSoftmaxScores` layer that can be appended through the `postprocessing` parameter to convert
the output of a model with sigmoid output activation to the output that would be produced by an equivalent model with softmax activation.


### Unpacking CZANN/CZSEG files

The czmodel library offers functionality to unpack existing CZANN/CZSEG models. For a given .czann or .czseg model it is possible to extract the underlying ANN model to a specified folder and retrieve the corresponding meta-data as instances of the meta-data classes defined in the czmodel library.

For CZANN files:

```python
from czmodel.convert import DefaultConverter
from pathlib import Path

model_metadata, model_path = DefaultConverter().unpack_model(model_file='Path of the .czann file', target_dir=Path('Output Path'))
```

For CZSEG files:

```python
from czmodel.convert import LegacyConverter
from pathlib import Path

model_metadata, model_path = LegacyConverter().unpack_model(model_file='Path of the .czseg file', target_dir=Path('Output Path'))
```


## CZANN Model Specification

This section specifies the requirements for an artificial neural network (ANN) model and the additionally required metadata to enable execution of the model inside the ZEN Intellesis infrastructure starting with ZEN blue >=3.2 and ZEN Core >3.0.

The model format currently allows to bundle models for semantic segmentation, instance segmentation, object detection, classification and regression and is defined as a ZIP archive with the file extension .czann containing the following files with the respective filenames:

- JSON Meta data file. (filename: model.json)
- Model in ONNX/TensorFlow SavedModel format. In case of  SavedModel format the folder representing the model must be zipped to a single file. (filename: model.model)
- Optionally: A license file for the contained model. (filename: license.txt)

The meta data file must comply with the following specification:

```json
{
    "$schema": "http://iglucentral.com/schemas/com.snowplowanalytics.self-desc/schema/jsonschema/1-0-0#",
    "$id": "http://127.0.0.1/model_format.schema.json",
    "title": "Exchange format for ANN models",
    "description": "A format that defines the meta information for exchanging ANN models. Any future versions of this specification should be evaluated through https://docs.snowplowanalytics.com/docs/pipeline-components-and-applications/iglu/igluctl-0-7-2/#lint-1 with --skip-checks numericMinMax,stringLength,optionalNull and https://www.json-buddy.com/json-schema-analyzer.htm.",
    "type": "object",
    "self": {
        "vendor": "com.zeiss",
        "name": "model-format",
        "format": "jsonschema",
        "version": "1-0-0"
    },
    "properties": {
        "Id": {
            "description": "Universally unique identifier of 128 bits for the model.",
            "type": "string"
        },
        "Type": {
            "description": "The type of problem addressed by the model.",
            "type": "string",
            "enum": ["SingleClassInstanceSegmentation", "MultiClassInstanceSegmentation", "SingleClassSemanticSegmentation", "MultiClassSemanticSegmentation", "SingleClassClassification", "MultiClassClassification", "ObjectDetection", "Regression"]
        },
        "MinOverlap": {
            "description": "The minimum overlap of tiles for each dimension in pixels. Must be divisible by two. In tiling strategies that consider tile borders instead of overlaps the minimum overlap is twice the border size.",
            "type": "array",
            "items": {
                "description": "The overlap of a single spatial dimension",
                "type": "integer",
                "minimum": 0
            },
            "minItems": 1
        },
        "Classes": {
            "description": "The class names corresponding to the last output dimension of the prediction. If the last dimension of the prediction has shape n the provided list must be of length n",
            "type": "array",
            "items": {
                "description": "A name describing a class for segmentation and classification tasks",
                "type": "string"
            },
            "minItems": 2
        },
        "ModelName": {
            "description": "The name of exported neural network model in ONNX (file) or TensorFlow SavedModel (folder) format in the same ZIP archive as the meta data file. In the case of ONNX the model must use ONNX opset version 12. In the case of TensorFlow SavedModel all operations in the model must be supported by TensorFlow 2.0.0. The model must contain exactly one input node which must comply with the input shape defined in the InputShape parameter and must have a batch dimension as its first dimension that is either 1 or undefined.",
            "type": "string"
        },
        "InputShape": {
            "description": "The shape of an input image. A typical 2D model has an input of shape [h, w, c] where h and w are the spatial dimensions and c is the number of channels. A 3D model is expected to have an input shape of [z, h, w, c] that contains an additional dimension z which represents the third spatial dimension. The batch dimension is not specified here. The input of the model must be of type float32 in the range [0..1].",
            "type": "array",
            "items": {
                "description": "The size of a single dimension",
                "type": "integer",
                "minimum": 1
            },
            "minItems": 3,
            "maxItems": 4
        },
        "OutputShape": {
            "description": "The shape of the output image. A typical 2D model has an input of shape [h, w, c] where h and w are the spatial dimensions and c is the number of classes. A 3D model is expected to have an input shape of [z, h, w, c] that contains an additional dimension z which represents the third spatial dimension. The batch dimension is not specified here. If the output of the model represents an image, it must be of type float32 in the range [0..1].",
            "type": "array",
            "items": {
                "description": "The size of a single dimension",
                "type": "integer",
                "minimum": 1
            },
            "minItems": 3,
            "maxItems": 4
        }
    },
    "required": ["Id", "Type", "InputShape", "OutputShape"]
}
```

Json files can contain escape sequences and \\-characters in paths must be escaped with \\\\.

The following code snippet shows an example for a valid metadata file:

For single-class semantic segmentation:

```json
{
  "Id": "b511d295-91ff-46ca-bb60-b2e26c393809",
  "Type": "SingleClassSemanticSegmentation",
  "Classes": ["class1", "class2", "class3"],
  "InputShape": [1024, 1024, 3],
  "OutputShape": [1024, 1024, 5]
}
```

For regression:

```json
{
  "Id": "064587eb-d5a1-4434-82fc-2fbc9f5871f9",
  "Type": "Regression",
  "InputShape": [1024, 1024, 3],
  "OutputShape": [1024, 1024, 3]
}
```

## Disclaimer

The libary and the notebook are free to use for everybody. Carl Zeiss Microscopy GmbH undertakes no warranty concerning the use of those tools. Use them at your own risk.

**By using any of those examples you agree to this disclaimer.**

Version: 2022.29.06

Copyright (c) 2022 Carl Zeiss AG, Germany. All Rights Reserved.
