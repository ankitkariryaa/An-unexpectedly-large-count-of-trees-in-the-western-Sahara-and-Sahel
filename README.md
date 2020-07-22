# An unexpectedly large count of trees in the western Sahara and Sahel
This repository contains the neural network model (UNet) and other essential codes for segmenting trees in Sahara and Sahel. The code was written by Ankit Kariryaa (Kariryaa AT uni-bremen DOT de) in 2018. Please contact him if you have any questions.

## Setup and Installation
See [INSTALL](./INSTALL.md).

## Structure

The code is structured in Jupyter notebooks available in the noteooks/ folder. Each notebook contains a considerable part of the pipeline. The notebooks are supported with core libraries available in the notebooks/core directory. Input, output paths and other configurations for each notebook must be declare notebooks/config directory. Follow these four steps for training a model from scratch and analyzing images.


### Step 1: Data preparation - [Preprocessing.ipynb](notebooks/1-Preprocessing.ipynb)
A part of the satellite images should be annotated with the trees (or other objects of interest), the areas that are annotated should be separately marked and stored as shapefiles. Once the dataset is ready, declare the relevants paths in the configuration. Copy notebooks/configTemplate/ directory into notebooks/config/ and declare the input and output paths and other relevant configurations in notebooks/config/Preprocessing.py file.
Once the dataset is prepared and the requied paths are declared in notebooks/config/, run the first notebook notebooks/1-Preprocessing.ipynb to extact these areas and the respective trees as image files. The extracted images will be written in a separate folder as defined in the config. 

### Step 2: Model training - [UNetTraining.ipynb](notebooks/2-UNetTraining.ipynb)  
Train the UNet model with the extracted images through the UNetTraining.ipynb notebook. Declare the relevant configuration in notebooks/config/UNetTraining.py. Use Auxiliary-1-UNetEvaluation.ipynb to evaluate the performance of the model on an independent dataset.

### Step 3: Analyzing images - [RasterAnalysis.ipynb](notebooks/3-RasterAnalysis.ipynb)
Use the trained model and analyze images with the help of RasterAnalysis.ipynb notebook. The images to be analyzed can be split into smaller images with the help of Auxiliary-2-SplitRasterToAnalyse.ipynb notebook, if the machine doesn't have enough memory to handle large Raster files. 

### Step 4: Postprocessing; Compression and Conversion to polygons - [notebooks/scripts/](notebooks/scripts/)
Use the scripts in the notebooks/scripts folder for compressing the generated images and for converting raster data to a vector polygon layer.
   



