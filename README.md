Code behind paper titled ___Quantum classification for synthetic aperture radar___.

Read the paper here: [https://doi.org/10.1117/12.3016462](https://doi.org/10.1117/12.3016462)

quantum_convolution.py performs the static quantum convolutions on the images as a preprocessing step and saves them in a separate folder. qcnn.py then trains and tests the CNN model on this preprocessed data. It uses the QuanvDataset class from quanv_dataset.py to read the preprocessed images.
