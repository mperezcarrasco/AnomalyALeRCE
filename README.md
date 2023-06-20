# Anomaly Detector for ALeRCE broker 🤖🌟

We present the anomaly detector methodology used in our article [ALeRCE Broker](https://alerce.science/)

### Preprocess data 

To preprocess the data used in the paper, download the folder [data_raw](https://drive.google.com/drive/folders/1z4qdQI60V82AmlwS_1Yxlqb1Vv1w04bB?usp=sharing) and use the `./preprocessing` module.

`cd preprocess`


`python3 main.py --labels_file --features_file --features_list`


### Train models

To reproduce paper results, download the folder [data](https://drive.google.com/drive/folders/1z4qdQI60V82AmlwS_1Yxlqb1Vv1w04bB?usp=sharing). 

`cd src`


`python3 launch_experiment.py --model --hierClass --lr --z_dim`

Models available are Autoencoder (ae), Variational Autoencoder (vae), Deep Suppor Vector Data Description (deepsvdd) and our proposed method Multi-Class Deep SVDD (classvdd).