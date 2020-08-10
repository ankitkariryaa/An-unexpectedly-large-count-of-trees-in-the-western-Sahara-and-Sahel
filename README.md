# An unexpectedly large count of trees in the western Sahara and Sahel
This repository contains the neural network model (UNet) and other essential codes for segmenting trees in Sahara and Sahel. The code was written by Ankit Kariryaa (Kariryaa AT uni-bremen DOT de) in 2018. Please contact him if you have any questions.

## Setup and Installation
See [INSTALL](./INSTALL.md).

## Structure
The code is structured in Jupyter notebooks available in the noteooks/ folder. Each notebook contains a considerable part of the pipeline and they are supported with core libraries available in the notebooks/core directory. Input, output paths and other configurations for each notebook must be declared in the notebooks/config/ directory. Please follow these four steps for training a UNet model and for analyzing images using the trained UNet model.


### Step 1: Data preparation - [Preprocessing.ipynb](notebooks/1-Preprocessing.ipynb)
The data has two main components, the satellite images and the label of trees/objects in those images. A part of the satellite images should be annotated with the trees (or other objects of interest), the areas that are annotated should be separately marked and stored as shapefiles. The SampleAnnotations/ directory contains examples of the format of shapefiles containing the labelled areas and the object polygons. The shapefiles should include a valid id column and no other attribute columns.
Once the dataset is ready, start by declaring the relevant paths in the configuration. Copy notebooks/configTemplate/ directory into notebooks/config/ and declare the input paths and other relevant configurations in notebooks/config/Preprocessing.py file. After declaring the required paths, run the first notebook notebooks/1-Preprocessing.ipynb to extract these areas with the contained object polygons as image files. The extracted images will be written in a separate folder in the defined path.

### Step 2: Model training - [UNetTraining.ipynb](notebooks/2-UNetTraining.ipynb)
Train the UNet model with the extracted images using the UNetTraining.ipynb notebook. Declare the relevant configuration in notebooks/config/UNetTraining.py. In case you use an independent test set, you can use Auxiliary-1-UNetEvaluation.ipynb to evaluate the performance of the model. Step-1 data preparation can also be used to extract the test set.

### Step 3: Analyzing images - [RasterAnalysis.ipynb](notebooks/3-RasterAnalysis.ipynb)
Next, use the trained model to analyze images using RasterAnalysis.ipynb notebook. The path to the trained model and satellite images should be declared in the notebooks/config/RasterAnalysis.py. The images to be analyzed can be split into smaller images with the help of Auxiliary-2-SplitRasterToAnalyse.ipynb notebook if the machine doesn't have enough memory to handle large Raster files.

### Step 4: Postprocessing; Compression and Conversion to polygons - [notebooks/scripts/](notebooks/scripts/)
Use the scripts in the notebooks/scripts folder for compressing the generated images and for converting raster data to a vector polygon layer.
   

## A note on the data source
This code is customized for the data that was available to us. It relies on satellite images with two channels and assumes that the two channels are stored in independent files and consequently read independently. The effect of the data source is evident in the code in notebooks and core libraries. In case your data is available in a single file (maybe with multiple channels), then the notebooks and core libraries need to be adapted accordingly. 