# LittleBigData
Python code to perform deep learning analysis on data sets with limited sample size and high dimensional data.

## Files:
 - LittleBigData.py, contains all the functions for analysis 
 
 - LittleBigData_main_Drive.ipynb, a Jupyter Notebook that can be uploaded into Google Drive and used with the Colaboratory app
 
 - LittleBigData_main_Linux.ipynb, a Jupyter Notebook that can be run natively on any OS system, but only Linux OS will show the Tensorboard

## Functions: 
*Preproc(x, w, Xstart, Xend, Wstart, Wend, trainfiledir, valfiledir, classtargets, header, testfiledir = None)*
 - **x** (integer), how many rows each input contains
 
 - **w** (integer), how many columns each input contains
 
 - **Xstart**, Xend, Wstart, Wend (integer), which row/column to start/end analysis, used for cropping
 
 - **trainfiledir**, **valfiledir**, **testfiledir** (string), file path to location of .csv files of data for training, validation, and testing (if provided)
 
 - **classtargets** (list), the marker in the file name that denotes ground truth label
 
 - **header** (interger), number of rows to skip in the .csv file of data
 
***

*Modeltrain(Xtrain, Ytrain, Xval, Yval, mname, net = '5FCN', ep = 80, bsize = 50)*
 - **Xtrain**, **Xval** (array), 4D tensor containing data with the first dimension representing inputs, second representing rows for each input, third represnting columns for each input, and fourth representing channels for each input (usually only 1).
 
 - **Ytrain**, **Yval** (array), one-hot vector containing ground truth labels for each input.
 
 - **mname** (string), name of the model for tensorboard and to save progress.
 
 - **net** (string or specified architecture), can choose '3FCN', '5FCN', or 'AlexNet' to select prewritten architectures or write your own and specify the variable name here. Default is 5FCN.
 
 - **ep** (integer), number of epochs. Defaul 80.
 
 - **bsize** (integer), batch size. Default 50.
 
 - - **OS** (string), either 'Colab' or 'Linux' used to show Tensorboard. If running on Windows or Mac, do not specify anything, defaults to None
 
***

*DeepDiscovery(Xval, Yval, classnames, n, modeldir = None, mname = None, model = None, net = '5FCN')*
 - **classnames** (list), character strings representing class names in the order of representation in the one-hot vector.
 
 - **n** (integer), number of inputs per sample.
 
 - **modeldir** (string) and **mname** (string), file path to the saved model and the model name to load in weights. Must either include this or model.
 
 - **model** (trained model), variable for the model already loaded into the jupyter notebook.
 
***

*GestaltDL(Xval, Yval, valnames, classnames, n, perc,  Xtest = None, Ytest = None, testnames = None, modeldir = None, mname = None, model = None, net = '5FCN')*
 - **valnames**/**testnames** (list), names of the samples in the validation/testing set (generally use file names).
 
 - **perc** (integer), number between 0 and 100 of the percentile of interest for identifying strong signals (i.e. if looking for the top 10% of signals, choose 90).
 
 ## Running Little, Big Data
 Two main Jupyter Notebooks are included: "LittleBigData_main_Drive.ipynb" and "LittleBigData_main_Linux.ipynb". If using Colaboratory with Google Drive, use the first file. If running python natively on your computer, use the second. 
 
 The Colaboratory script is set up to allow easy syncing with Google Drive and includes all the commands needed to access data stored in a folder on your Drive that contains the data. That said, Colaboratory updates frequently so this file may require some troubleshooting. 
 
 The Linux script is written and tested using the Ubuntu OS. That said, it can work on a Windows or Mac machine as long as the OS command in Model Training is specified as "None" or left blank (and defaults to "None"). The advantage to running model training on Linux or Colaboratory is to view the Tensorboard which shows graphical summaries of the training process, however it is not necessary to train the model.
 
 ### To Use Little, Big Data in Colaboratory with Google Drive
 Colaboratory is a Google Drive-based service that allows you to run Python on the cloud using Google's GPUs. It is a great resource for those who do not have access to GPU computing power or to run smaller models that do not take as long to train, however because it is cloud-based and shares resources with other users, it may not be the most reliable. However, it does provide an easy way to analyze data and you can train deep neural networks on it. 
 
 The structure of a Colaboratory Notebook is essentially the same as a Jupyter Notebook, which mitigates most formatting issues when moving between the two. Because of this, you can download the "LittleBigData_main_Drive.ipynb" from this repository and upload it to your Google Drive account without issue. You may have to link Colaboratory with your Google Drive account the first time to open up the file, but once you add the Colaboratory app you will not need to do so again.
 
 The first step in using Little, Big Data in Colaboratory with Google Drive is to upload your data onto your drive. Samples for your two (Training, Validation) or three (Training, Validation, Testing) sets should be stored in separate folders. Additionally, it is suggested to choose a specific location to save your trained model in. 
 
 Once your data are organized into three folders, you have to include the "LittleBigData.py" file that contains all of the functions into your working directory. Ideally, your Colaboratory Notebook, the LittleBigData.py file, and the folders for your data/model would be organized together into one folder, but it is not necessary.
 
 you will need to mount your Google Drive. This is done with the following command: 
 
 `drive.mount('/content/drive')`
 
 Which is imported from `google.colab`. These two lines of code are included in the Colaboratory Notebook. When you mount your drive, you willget a message that says "Go to this URL in a browswer:" followed by a hyperlink. Clicking on that link allows you to select which Google account to sync. After you click the appropriate account, it will give you an authorization code to copy and paste into the text box below the URL in the Colaboratory notebook. After entering the authorization code and hitting "Enter" you should get the message "Mounted at /content/drive".
 
 Now you are ready to complete Little, Big Data in the same way as if you were running it natively on your computer. 
 
 ### To Use Little, Big Data Natively
 If you have Python and Jupyter Notebook installed on your computer and have enough processing power to train a DNN, you can use the "LittleBigData_main_Linux.ipynb". All that is necessary to do is download the Jupyter Notebook, LittleBigData.py, and store your data in two or three folders (Training, Validation, and Testing Set, if provided). This script was written for Linux machines, but can run on Windows/Mac as long as the OS is not specified or specified as "None" in the Modeltrain command. 
 
 If you have Python and Jupyter Notebook installed on your computer, but do not have the ability to train a DNN, you can also do the first half of the analysis on Colaboratory and the second half natively. If you choose to to it that way, it is suggested that after preprocessing your data, you save it as ".npy" files on the Drive and download them so you do not have to preprocess your data twice. Saving and loading data using that file extension is explained in the Jupyter/Colaboratory notebooks.
 
 To natively postprocess a model you trained on Colaboratory using Little, Big Data you need to make sure you have access to your data files (either originals or the ".npy" extension), the model files (one with the extension ".data-" followed by some number of some other number, ".index" and ".meta"), and the LittleBigData.py file. You may have to change directories at different points to do all this, but that is done simply with the `os.chdir()` command. The file path will go in the parenthesis and to check which directory you are in you can use `os.getcwd()` leaving the parentheses enpty or to list the files in the current directory, use `os.listdir()` leaving the parentheses empty.
 
 ## To Use with Juice/GC-IMS Dataset
 A set of GC-IMS data is provided to work through Little, Big Data. In this dataset there are three classes ("A", "B", and "C") which correspond to samples of three types of juice ("Apple Juice", "Tropicana Orange Juice", and "Heritage Orange Juice"). The outputs are stored in .csv files containing 2000 rows (retention time) and 4500 columns (drift time). There are 20 samples per class, resulting in 60 samples total. The data is not separated into three folders for training, validation, and testing. Each .csv file is 69 MB large for a total of 4.14 GB.
 
 This dataset is provided by Emily Stark under the supervision of Dr. Elan Barenholtz and Dr. William Hahn of the MPCR Lab at Florida Atlantic University in Boca Raton, FL and was generated by Alfian Nur Wicaksono under the supervision of Dr. James Covington of the Biomedical Sensing Lab at the University of Warwick in Coventry, UK.
 
 This data can be downloaded [here](https://drive.google.com/drive/folders/1tDQxmCkciojeBIBSVIlq59tLq84mmXvV?usp=sharing)
