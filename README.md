# estimate_age_gender_cnn

An experiment was conducted with a view to perform age and gender estimation on over 500,000+
static face images of celebrities available scraped from IMDB and Wikipedia (https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/).

### Data Prep
We downloaded the IMDB-Wiki Dataset available as Tar file formats with images from 460,723 IMDB and 62,328 images from Wikipedia. A python code snippet available in download_check_dataset.ipynb​ ​was used to scrape the data from the website and extract the necessary files.

Filtering part is done in ​create_db.py​, ​here we provide the imdb.mat or wiki.mat as input. Basically, we delete all rows from .mat file with:
  1. NaN values in gender label
  2. face_score less than min_score (configurable; default = 1.0)
  3. NaN and negative values in second_face_score
  4. age labels less equal than 0 and greater than equal to 100.

Further the images are resized to 64x64 (configurable). The filtered values are stored in imdb_db.mat and wiki_db.mat files.

### Neural Network Creation and Training

we used Keras to implement Wide Residual Networks. The neural network used for this project comprised of 16 convolutional layers and feature maps per layer in [16,128,256,512] i.e. 16 units deep and [16,128,256,512] units wide. Also, we use batch normalization​ to normalize the input layer as well as hidden layers by adjusting mean and scaling of the activations (allowing the network use higher learning rate without vanishing or exploding gradients). ​ReLU is the activation function used to introduce non-linearity into the model. Average Pooling is the pooling method used to reduce the spatial size of the network.
The output of the pooling layer is flattened and fed to the two fully connected layers which predict the provide age and gender related labels as output.

creation of neural network related code is available in ​wide_resnet.py

we trained the model on IMDB filtered images (171,852). ​​The codes for training the model are available in ​train.py​ and ​training_model.ipynb. "training_model.ipynb" calls the train.py script, this was done as we wanted to use Google Colab's GPUs for training the model.

Prediction related code can be found in predict.ipynb file.

The predictions and model details can be found in results folder.



