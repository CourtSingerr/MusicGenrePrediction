# Music Genre Classification with Deep Learning

This repository contains the code and documentation for a deep learning project focused on classifying music genres from audio features using convolutional neural networks (CNNs). Using the GTZAN dataset as a benchmark, this project compares different audio feature representations (Log-Mel Spectrograms, MFCCs, and MFCCs with delta features) and employs hyperparameter tuning to improve model performance.

### Overview
- Dataset: GTZAN Genre Collection
- Genres: 10 categories (Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, Rock)
- Goal: Predict the genre of a music track segment using a CNN trained on extracted audio features.

#### Key Steps
1.	Data Preprocessing & Segmentation: Split 30-second audio tracks into shorter segments (e.g., 6-second clips) to increase the number of training examples.
2.	Feature Extraction:
	- Log-Mel Spectrograms
	- MFCC (Mel-Frequency Cepstral Coefficients)
	- MFCC + Deltas

	Normalize features and apply one-hot encoding for labels.

   Model Architecture:
	A baseline CNN model:
	- 3 Convolutional blocks (Conv2D + BatchNorm + MaxPooling)
	- Dense layers with dropout
 	- Softmax output for 10-class classification
4.	Comparative Evaluation:
Trained the same baseline model on each feature representation to identify which performs best. MFCCs emerged as the strongest features.
	5.	Hyperparameter Tuning:
Used Keras Tuner (RandomSearch) to fine-tune hyperparameters (filters, dropout, learning rate) on the MFCC dataset, achieving improved accuracy.
	6.	Evaluation & Results:
Final model accuracy of ~64.5% on the test set. Explored per-class performance, confusion matrices, and identified areas for improvement.
