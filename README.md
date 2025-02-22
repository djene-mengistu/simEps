# Semi-Supervised Defect Segmentation with Pairwise Similarity Map Consistency and Ensemble-Based Cross-Pseudo Labels (simEps)
In this study, we propose a novel method based on pairwise similarity map consistency with ensemble-based cross-pseudo labels for semisupervised defect segmentation that uses limited labeled samples while exploiting additional label-free samples. The proposed approach uses three network branches that are regularized by pairwise similarity map consistency, and each of them is supervised by the pseudo labels generated by ensemble of predictions of the other two networks for the unlabeled samples. The proposed method achieved significant performance improvement over the baseline of learning only from the labeled images and the current stateof-the-art semi-supervised methods.\
The overall architecture of the proposed method with three parallel netwroks is presented as follows:
<p align="center">
<img src="/ext_data/main-architecture.jpg" width="60%" height="50%">
</p>

**Fig. 1:** The overall framework of the proposed method. The labeled and unlabeled samples are fed to the networks to generate multi-scale outputs, P<sup>e</sup>, P<sup>d</sup> and P<sup>o</sup>. **(a)** The supervised segmentation loss ($\ell$<sub>seg</sub>) is computed on the labeled samples using the ground truth (GT) and the network outputs. **(b)** For the unlabeled images, the pseudo-supervision loss ($\ell$<sub>ps</sub>) is computed using the pseudo labels (pl) generated from ensemble of predictions on the two networks and the output of the third network. The network outputs (P<sup>e</sup>, P<sup>d</sup> and P<sup>o</sup>) are utilized to compute the pairwise similarity map consistency loss ($\ell$<sub>sim</sub>).

The pairwise similarity map consistecy loss among the parallel networks at three stages is illustrated in the following figure.
 
<p align="center">
<img src="/ext_data/pairwise_loss.jpg" width="60%" height="50%">
</p>

**Fig. 2:** Paiwrwise similarity map consistency loss computation.

# Full paper source:
You can read the details about the methods, implementation, and results from: (https://ieeexplore.ieee.org/document/9994033)

**Please cite ourwork as follows:**
```
@article{sime2022semi,
  title={Semi-Supervised Defect Segmentation with Pairwise Similarity Map Consistency and Ensemble-Based Cross-Pseudo Labels},
  author={Sime, Dejene M and Wang, Guotai and Zeng, Zhi and Wang, Wei and Peng, Bei},
  journal={IEEE Transactions on Industrial Informatics},
  year={2022},
  publisher={IEEE}
}
```
## Python >= 3.6
PyTorch >= 1.1.0
PyYAML, tqdm, tensorboardX
## Data Preparation
Download datasets. There are 3 datasets to download:
* NEU-SEG dataset from [NEU-seg](https://ieeexplore.ieee.org/document/8930292)
* DAGM dataset from [DAGM](https://www.kaggle.com/datasets/mhskjelvareid/dagm-2007-competition-dataset-optical-inspection)
* MT (Magnetic Tiles) dataset from [MTiles](https://www.kaggle.com/datasets/alex000kim/magnetic-tile-surface-defects)

Put downloaded data into the following directory structure:
* data/
    * NEU_data/ ... # raw data of NEU-Seg
    * DAGM_data/ ...# raw data of DAGM
    * MTiles_data/ ...# raw data of MTiles
## Code usage
The training files and settings for each compared network is presented in separate directory. Train each network and test from the presented directory.
To train the proposed **simEps** method run the following after setting hyperparameters such as labeled-ratio, iteration-per-epoch, consistency ramp length, and pair-wise-similarity loss coefficient.
```bash
python simEps_train.py
```

To test the performance of the proposed method:
```bash
run simEps_Testing.ipynb
```

To evaluate and visualize the pairwise similarirty map:
```bash
run simEps_evalaute.ipynb
```
Similarly, train the proposed method, **simEps**, for the other datasets from the indicated directories after setting appropriate hyper-parametres.
## Some results and visualization
The results of the proposed method compared with the supervised baseline is presented as follows:

 <p align="center">
<img src="/ext_data/com-baseline.JPG" width="40%" height="30%">
</p>

**Fig. 3:** Results from the proposed method vs. the supervised baseline on the NEU-Seg dataset.

<p align="center">
<img src="/ext_data/com_dlv3.JPG" width="40%" height="40%">
</p>

**Fig. 4:** Comparison of results using different baseline networks (DLV3+ and UNet).

The visualization of the segmetnation results for the baseline, selected semi-supervised methods and our proposed method with the NEU-Seg dataset is presented as follows.
 
<p align="center">
<img src="/ext_data/NEU-viz.jpg" width="50%" height="40%">
</p>

**Fig. 5:** Visualization of segmentation results on the NEU_seg dataset.

<p align="center">
<img src="/ext_data/pairwise_similarity_viz.jpg" width="60%" height="50%">
</p>

**Fig. 6:** Visualization of paiwrwise similarity map computed at multi-stages after training.

## Acknowledgment

This repo borrowed many implementations from [SSL4MIS](https://github.com/HiLab-git/SSL4MIS)

## Contact
For any issue please contact me at djene.mengistu@gmail.com
