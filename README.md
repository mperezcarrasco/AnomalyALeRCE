
 
<p align="center">
  <img  src="../presentation/images/anomaly_logo.png" alt="ALeRCE AD logo" width="100%">
</p>


## Anomaly Detector for ALeRCE broker

Welcome to the **ALeRCE anomaly detector** framework. This is the main repository where you can find all the resources associated to our article:  [Alert Classification for the ALeRCE Broker System: The Anomaly Detector](https://alerce.science/).

Our methodology has been published and is available through the [ALeRCE Broker](https://alerce.science/), a Chilean-led platform that processes the alert stream from the Zwicky Transient Facility (ZTF).


In our work we look for the best outliers detection algorithms that aim to find transient, periodic and stochastic  anomalous sources within the ZTF data stream. The **ALeRCE anomaly detector** framework consists of crossvalidating **six anomaly detection algorithms** for each of these three classes using the ALeRCE light curve features.


Following the ALeRCE taxonomy, we consider four transient subclasses, five stochastic subclasses, and five periodic subclasses. We evaluate each algorithm by considering each subclass as the anomaly class.

<h1><img src="https://alerce-science.s3.amazonaws.com/images/taxonomy.max-1600x900.png" alt="ALeRCE taxonomy" width="100%"><p></p></h1>

## About our work

We provide a Machine and Deep Learning-based framework for anomaly detection. Our methodology is inspired by the [ALeRCE's light curve (LC) classifier](https://iopscience.iop.org/article/10.3847/1538-3881/abd5c1) and follows a hierarchical approach. The light curves are categorized into three main classes: transient, stochastic, and periodic, at the top level. For each class, a distinct anomaly detection model is constructed, utilizing only information about the known objects (i.e., inliers) for training. During testing, in order to assign the light curve to one of the anomaly detectors and compute the anomaly score, we use the probabilities, as given by ALeRCE's LC classifier, indicating whether the light curve corresponds to a transient, stochastic, or periodic nature.

<details open><summary>Pipeline</summary>

<p align="center">
  <img  src="../presentation/images/pipeline.png" alt="ALeRCE taxonomy" width="100%">
</p>

**Figure 1.** Methodology for training and evaluation of the anomaly detection algorithms. We split the data into a training set and test set composed by 80% and 20% of the data, respectively The training set is subdivided into transient stochastic, and periodic data. For each of these classes, we choose each subclass as the outlier class. The outlier class is removed from the training set and added to the test set (TS2). Then, an anomaly detection algorithm is trained using the remaining objects of each of the classes, and is evaluated using TS2.

</details>

<details open><summary>Results</summary>

See the notebooks (`/presentation/notebooks/*`) for usage examples with these models.

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-jmd0{border-color:inherit;font-family:"Lucida Sans Unicode", "Lucida Grande", sans-serif !important;font-weight:bold;
  text-align:center;vertical-align:top}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
.tg .tg-yvi3{border-color:inherit;font-family:"Lucida Sans Unicode", "Lucida Grande", sans-serif !important;text-align:center;
  vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky"></th>
    <th class="tg-jmd0" colspan="4">Transient</th>
    <th class="tg-jmd0" colspan="5">Stochastic</th>
    <th class="tg-jmd0" colspan="5">Periodic</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-jmd0">Method</td>
    <td class="tg-jmd0">SLSN</td>
    <td class="tg-jmd0">SNII</td>
    <td class="tg-jmd0">SNIa</td>
    <td class="tg-jmd0">SNIbc</td>
    <td class="tg-jmd0">AGN</td>
    <td class="tg-jmd0">Blazar</td>
    <td class="tg-jmd0">CV/Nova</td>
    <td class="tg-jmd0">QSO</td>
    <td class="tg-jmd0">YSO</td>
    <td class="tg-jmd0">CEP</td>
    <td class="tg-jmd0">DSCT</td>
    <td class="tg-jmd0">E</td>
    <td class="tg-jmd0">RRL</td>
    <td class="tg-jmd0">LPV</td>
  </tr>
  <tr>
    <td class="tg-yvi3">IForest</td>
    <td class="tg-yvi3">0.640<br>±0.014</td>
    <td class="tg-yvi3">0.721<br>±0.021</td>
    <td class="tg-yvi3">0.428<br>±0.032 </td>
    <td class="tg-yvi3">0.490<br>±0.038</td>
    <td class="tg-yvi3">0.573<br>±0.017</td>
    <td class="tg-yvi3">0.710<br>±0.009</td>
    <td class="tg-yvi3">0.975<br>±0.001</td>
    <td class="tg-yvi3">0.468<br>±0.016</td>
    <td class="tg-yvi3">0.913<br>±0.003</td>
    <td class="tg-yvi3">0.359<br>±0.007</td>
    <td class="tg-yvi3">0.295<br>±0.012</td>
    <td class="tg-yvi3">0.469<br>±0.021</td>
    <td class="tg-yvi3">0.549<br>±0.033</td>
    <td class="tg-yvi3">0.971<br>±0.007</td>
  </tr>
  <tr>
    <td class="tg-yvi3">OCSVM</td>
    <td class="tg-yvi3">0.577<br>±0.014</td>
    <td class="tg-yvi3">0.587<br>±0.014</td>
    <td class="tg-yvi3">0.434<br>±0.021</td>
    <td class="tg-yvi3">0.492<br>±0.011</td>
    <td class="tg-yvi3">0.532<br>±0.008</td>
    <td class="tg-yvi3">0.443<br>±0.002</td>
    <td class="tg-yvi3">0.909<br>±0.001</td>
    <td class="tg-jmd0">0.517<br>±0.005</td>
    <td class="tg-yvi3">0.792<br>±0.005</td>
    <td class="tg-yvi3">0.432<br>±0.004</td>
    <td class="tg-yvi3">0.557<br>±0.005</td>
    <td class="tg-yvi3">0.555<br>±0.003</td>
    <td class="tg-yvi3">0.539<br>±0.004</td>
    <td class="tg-yvi3">0.943<br>±0.001</td>
  </tr>
  <tr>
    <td class="tg-yvi3">AE</td>
    <td class="tg-jmd0">0.736<br>±0.022</td>
    <td class="tg-yvi3">0.807<br>±0.021</td>
    <td class="tg-yvi3">0.438<br>±0.015</td>
    <td class="tg-yvi3">0.537<br>±0.019</td>
    <td class="tg-jmd0">0.701<br>±0.010</td>
    <td class="tg-jmd0">0.762<br>±0.006</td>
    <td class="tg-jmd0">0.980<br>±0.016</td>
    <td class="tg-yvi3">0.443<br>±0.004</td>
    <td class="tg-jmd0">0.990<br>±0.001</td>
    <td class="tg-yvi3">0.564<br>±0.024</td>
    <td class="tg-yvi3">0.367<br>±0.015</td>
    <td class="tg-yvi3">0.864<br>±0.009</td>
    <td class="tg-yvi3">0.907<br>±0.015</td>
    <td class="tg-jmd0">0.996<br>±0.000</td>
  </tr>
  <tr>
    <td class="tg-yvi3">VAE</td>
    <td class="tg-yvi3">0.669<br>±0.015 </td>
    <td class="tg-yvi3">0.690<br>±0.023</td>
    <td class="tg-yvi3">0.404<br>±0.018</td>
    <td class="tg-yvi3">0.522<br>±0.025</td>
    <td class="tg-yvi3">0.596<br>±0.007</td>
    <td class="tg-yvi3">0.597<br>±0.010</td>
    <td class="tg-yvi3">0.849<br>±0.028</td>
    <td class="tg-yvi3">0.500<br>±0.009</td>
    <td class="tg-yvi3">0.795<br>±0.009</td>
    <td class="tg-yvi3">0.442<br>±0.010</td>
    <td class="tg-yvi3">0.417<br>±0.007</td>
    <td class="tg-yvi3">0.561<br>±0.007</td>
    <td class="tg-yvi3">0.451<br>±0.006</td>
    <td class="tg-yvi3">0.936<br>±0.007</td>
  </tr>
  <tr>
    <td class="tg-yvi3">Deep SVDD</td>
    <td class="tg-yvi3">0.644<br>±0.043</td>
    <td class="tg-yvi3">0.690<br>±0.043</td>
    <td class="tg-yvi3">0.475<br>±0.040</td>
    <td class="tg-yvi3">0.507<br>±0.040</td>
    <td class="tg-yvi3">0.496<br>±0.025</td>
    <td class="tg-yvi3">0.607<br>±0.044</td>
    <td class="tg-yvi3">0.932<br>±0.015</td>
    <td class="tg-yvi3">0.411<br>±0.008</td>
    <td class="tg-yvi3">0.901<br>±0.022</td>
    <td class="tg-yvi3">0.707<br>±0.027</td>
    <td class="tg-yvi3">0.482<br>±0.054</td>
    <td class="tg-yvi3">0.636<br>±0.055</td>
    <td class="tg-yvi3">0.774<br>±0.068</td>
    <td class="tg-yvi3">0.785<br>±0.025</td>
  </tr>
  <tr>
    <td class="tg-yvi3">MCDSVDD<br>(Ours)</td>
    <td class="tg-yvi3">0.686<br>±0.051</td>
    <td class="tg-jmd0">0.828<br>±0.024</td>
    <td class="tg-jmd0">0.624<br>±0.039</td>
    <td class="tg-jmd0">0.584<br>±0.032</td>
    <td class="tg-yvi3">0.706<br>±0.069</td>
    <td class="tg-yvi3">0.512<br>±0.113</td>
    <td class="tg-yvi3">0.770<br>±0.127</td>
    <td class="tg-yvi3">0.483<br>±0.080</td>
    <td class="tg-yvi3">0.854<br>±0.041</td>
    <td class="tg-jmd0">0.858<br>±0.025</td>
    <td class="tg-jmd0">0.819<br>±0.015</td>
    <td class="tg-jmd0">0.945<br>±0.006</td>
    <td class="tg-jmd0">0.953<br>±0.003</td>
    <td class="tg-yvi3">0.953<br>±0.008</td>
  </tr>
</tbody>
</table>

**Table 1.** Evaluation of the performance of each model when applied to each of the ALeRCE top level taxonomy (transient, stochastic, periodic). Each row represents a different outlier detection algorithm, and each column represents the subclass considered as outlier. The performance is evaluated using the **cross–validation AUROC scores**.

As shown in Table 1, the best performance was achieved for transient and periodic sources using a modified version of the Deep Support Vector Data Description (MCDSVDD) neural network. However, for stochastic sources, the best results were obtained by calculating the reconstruction error of an autoencoder (AE) neural network.

</details>


## Features
- Six anomaly detection algorithms were implemented in order to compare their performances in finding outliers (`/src/models/*`)
  - Isolation Forest (`/src/models/IForest.py`)
  - One-class Support Vector Machine (`/src/models/*`)
  - Autoencoder (`/src/models/Autoencoder.py`)
  - Variational Autoencoder (`/src/models/VariationalAutoencoder.py`)
  - Deep Support Vector Data Description (`/src/models/DeepSVDD.py`)
  - Multi-Class Deep SVDD (`/src/models/ClasSVDD.py`)
  - Variational Deep Embedding (`/src/models/VaDE.py`)
-  A new scheme of training and evaluation methodology
-  Validated framework in a real-world scenario by selecting the 10 sources with the highest outlier score per each of the 15 classes predicted by the ALeRCE light curve classifier (`presentation/notebooks/Results analysis.ipynb`)
- Methods trained using the ZTF alert stream and benefit from the ALeRCE LC classifier 
- Predefined experiments to reproduce publication results (`presentation/experiments/*`)
- Data preprocessing, saving and reading (`/src/preprocessing/*`)
- Dockerfile and scripts for building (`build_container.sh`) and run (`run_container.sh`) the Anomaly Detector container



## Implementation tree
```
📦AnomalyALeRCE
 ┣ 📂data
 ┃ ┣ 📜 get_data.sh: download data (raw and preprocessed) from gdrive
 ┃ ┗ 📜 README.md: instructions how to use get_data.sh 
 ┣ 📂 experiments: experiments folder
 ┃ ┗ 📜 launch_experiments.sh: script for running all the published version of the experiments
 ┣ 📂 presentation: everything that depends on the model code (i.e., experiments, plots and figures)
 ┃ ┣ 📂 figures
 ┃ ┗ 📂 notebooks: util notebooks to visualize and analyze the results
 ┣ 📂 src: Model source code
 ┃ ┗ 📂 preprocessing: functions related to data manipulation
 ┃ ┃ ┣ 📜 ALeRCE_LC.py: data preparation functions inspired on ALeRCE LC classifier
 ┃ ┃ ┣ 📜 create_dataloaders.py: create the dataloaders for the ML and DL models
 ┃ ┃ ┣ 📜 ALeRCE_LC.py: data preparation functions inspired on ALeRCE LC classifier
 ┃ ┃ ┣ 📜 data_utils.py: general functions to standardize, among others
 ┃ ┃ ┗ 📜 main.py: main script containing functions to format data
 ┃ ┗ 📂 models: Anomaly Detector models architectures
 ┃ ┃ ┣ 📜 main.py: build the network with the selected classifier
 ┃ ┃ ┣ 📜 Autoencoder.py: Autoencoder
 ┃ ┃ ┣ 📜 ClasSVDD.py: Multi-Class Deep SVDD (ours)
 ┃ ┃ ┣ 📜 DeepSVDD.py: Deep Support Vector Data Description
 ┃ ┃ ┗ 📜 VariationalAutoencoder.py: Variational autoencoder
 ┃ ┗ 📂 utils.py: universal functions to use on different modules
 ┃ ┣ 📜 train.py: functions related to the training stage of the models
 ┃ ┗ 📜 evaluate.py: functions related to the inference stage of the models
 ┣ 📜 .gitignore: files that should not be considered during a GitHub push
 ┣ 📜 .dockerignore: files to exclude when building a Docker container.
 ┣ 📜 build_container.sh: script to build the Anomaly Detector Docker image
 ┣ 📜 run_container.sh: script to run the Anomaly Detector Docker image (up container)
 ┣ 📜 Dockerfile: Docker image definition
 ┣ 📜 requirements.txt: python dependencies
 ┗ 📜 README.md: what you are currently reading
 ```

## Setup the enviroment
The easiest way to setup the environment is by using [Docker](https://docs.docker.com/get-docker/), since it provides a **kernel-isolated** and **identical environment** to the one used by the authors. 

The `Dockerfile` contains all the configuration for running the Anomaly Detector framework. No need to touch it, `build_container.sh` and `run_container.sh` make the work for you.


The first step is to build the container,
```bash
  bash build_container.sh
```
It creates a "virtual machine", named `anomalydetector`, containing all the dependencies such as python, tensorflow, among others. 

The next and final step is running the Anomaly Detector container,
```
  bash run_container.sh
```
The above script looks for the container named `anomalydetector` and run it on top of [your kernel](https://www.techtarget.com/searchdatacenter/definition/kernel#:~:text=The%20kernel%20is%20the%20essentialsystems%2C%20device%20control%20and%20networking.). Automatically, the script recognizes if there are GPUs, making them visible inside the container.

By default the `run_container.sh` script opens the ports `8888` and `6006` for **jupyter notebook** and [**tensorboard**](https://github.com/cridonoso/tensorboard_tutorials), resepectively. To run them, use the usal commands but adding the following lines:

For Jupyter Notebook 
```
jupyter notebook --ip 0.0.0.0
```
For Tensorboard
```
tensorboard --logdir <my-logs-folder> --host 0.0.0.0
```
Finally, **if you do not want to use Docker** the `requirements.txt` file contains all the packages needed to run ASTROMER. Use `pip install -r requirements.txt` on your local python to install them.

### Dependencies 
* Python == 3.8.10
* Torch == 2.0.1
* Some more packages see `requirements.txt`

## Reproducibility
For reproducibility of our reported results run, we provide a script (`presentation/experiments/launch_experiments.bash`) that can be used to run all the models at once. To preprocess the data used in the paper, download the data following the next steps.  

```
src data
```

```
bash get_data.sh ztf-processed
```

## Using your own data 


```
cd preprocess
```


```
python3 main.py --labels_file --features_file --features_list
```


### Train models

To reproduce paper results, download the folder [data](https://drive.google.com/drive/folders/1z4qdQI60V82AmlwS_1Yxlqb1Vv1w04bB?usp=sharing). 

```
cd src
```


```
python3 launch_experiment.py --model --hierClass --lr --z_dim
```

Models (`--model`) available are Autoencoder (`ae`), Variational Autoencoder (`vae`), Deep Support Vector Data Description (`deepsvdd`) and our proposed method Multi-Class Deep SVDD (`classvdd`). Hierarchical classes (`--hierClass`) available are [`Transient`, `Stchastic`, `Periodic`].

## Contributing 🤝
Contributions are always welcome!


Issues and featuring can be directly published in this repository via [Pull Request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests). Look at [this tutorial](https://cridonoso.github.io/articles/github.html) for more information about pull requests.
