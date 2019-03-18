# autoencoder_for_cancer
autoencoder to compress methylation data and cancer cell classifier

This project consists of 3 main parts
1. loading data with `load_data` folder: 
  load methylation data from [GDC database](https://portal.gdc.cancer.gov)
2. preprocessor data with `preprocessor.py`: 
  remove features with `NA` data, remove low-variant features, and separate data of normal/cancer into separated files.
3. classifier with `model` folder: compress data and classify if it is cancer or not
  `autoencoder.py` train an `encoder` model used for compression
  `classifier.py` train a classifier for compressed data
  `predict.py` uses the trained classifier and encoder to classify new data
