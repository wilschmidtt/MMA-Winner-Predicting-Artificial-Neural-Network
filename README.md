Still in the process of creating the README for this project. The code is complete though, so the program can be run as long as all the necessary packages have been installed on your device and the included data files are downloaded.

# UFC Winner Predictor - Artificial Neural Network
Artificial neural network that predicts who will win a UFC contest between two fighters. The dataset used to train the neural network contains a vareity of statistics about fights dating back to as far as 1993 (when the UFC started collecting data), along with in-depth stats about each fighter. Once trained, the user can enter statistics about two fighters in question, all of which can be found at [ufcstats.com](http://www.ufcstats.com/fighter-details/f4c49976c75c5ab2), and the neural network will give you its prediction on who will win the fight, as well as the chance, expressed as a percentage, that it attributes to each fighter winning the contest. 

## Getting Started


### Prerequisites
* Anaconda (Python 3.7 Version)
  - [Anaconda Instillation Instructions](https://docs.anaconda.com/anaconda/install/)
  - Using Anaconda is optional, I simply chose to use it for the Spyder environment that is included

### Packages to Install
* pandas - software library for data manipulation and analysis
  - `conda install -c anaconda pandas`
* numpy - general-purpose array-processing package
  - `conda install -c anaconda numpy`
* keras - 

## Running the tests

## Authors
* **William Schmidt** - [Wil's LikedIn](https://www.linkedin.com/in/william-schmidt-152431168/)

## Acknowledgments

* Huge shoutout to Rajeev Warrier for scraping this data from the UFCstats website using Beautiful Soup! I did not use any of his preprocessed data files, simply the raw dataset, and then preprocessed it myself. I can't thank him enough for providing this dataset on Kaggle!
* [Rajeev's Kaggle Profile](https://www.kaggle.com/rajeevw)
* [Rajeev's Github Profile](https://github.com/WarrierRajeev?tab=repositories)
