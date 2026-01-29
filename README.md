This model decomposes the attention model and integrates various network features such as CNN, KAN, ResNet, and MLP, etc. It also automatically adjusts depth, width and the weights of each feature, achieving a super-set neural network capable of completing multiple tasks.

This model currently achieves Top1 accuracy rates of 98.5%, 91.7%, and 97.4% respectively on the ECG electrocardiogram data set, the cifar10 data set, and the EEG brainwave data set. The above text respectively covers the recognition tasks of high-noise one-dimensional data and two-dimensional images. The two data set folders contain the codes for the model, data preprocessing, training, and testing, as well as the trained parameter files.

The paper is currently under submission. My model outperforms the majority of the existing models in terms of accuracy and interpretability. The comparison with models such as LSTM, Wavlet_KAN, SOTA_EEG, CNN, MLP, and transformer is presented in the comparison test folder.
