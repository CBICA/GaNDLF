## 0.0.9

- Refactoring the training and inference code
- Added offline mechanism to generate padded images to improve training RAM requirements

## 0.0.8

- Pre-split training/validation data can now be provided
- Major code refactoring to make extensions easier
- Added a way to ignore a label during validation dice calculation
- Added more options for VGG
- Tests can now be run on GPU
- New scheduling options added

## 0.0.7

- New modality switch added for rad/path
- Class list can now be defined as a range
- Added option to train and infer on fused labels
- Rotation 90 and 180 augmentation added
- Cropping zero planes added for preprocessing
- Normalization options added
- Added option to save generated masks on validation and (if applicable) testing data

## 0.0.6

- Added PyVIPS support
- SubjectID-based split added

## 0.0.5

- 2D support added
- Pre-processing module added
  - Added option to threshold or clip the input image
- Code consolidation
- Added generic DenseNet
- Added option to switch between Uniform and Label samplers
- Added histopathology input (patch-based extraction)

## 0.0.4

- Added full image validation for generating loss and dice scores
- Nested cross-validation added
  - Collect statistics and plot them
- Weighted DICE computation for handling class imbalances in segmentation

## 0.0.3 

- Added detailed documentation
- Added MSE from Torch 
- Added option to parameterize model properties
  - Final convolution layer (softmax/sigmoid/none)
- Added option to resize input dataset
- Added new regression architecture (VGG)
- Version checking in config file

## 0.0.2

- More scheduling options
- Automatic mixed precision training is now enabled by default
- Subject-based shuffle for training queue construction is now enabled by default
- Single place to parse and pass around parameters to make training/inference API easier to handle
- Configuration file mechanism switched to YAML

## 0.0.1 (2020/08/25)

- First tag of GaNDLF
- Initial feature list:
  - Supports multiple
    - Deep Learning model architectures
    - Channels/modalities 
    - Prediction classes
  - Data augmentation
  - Built-in cross validation
