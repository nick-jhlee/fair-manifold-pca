# MbF-PCA (MMD-based Fair PCA)

This repository provides the implementation of MbF-PCA, described in the paper [*Fast and Efficient MMD-based Fair PCA via Optimization over Stiefel Manifold*](https://arxiv.org/abs/2109.11196), accepted to AAAI 2022, by Junghyun Lee, Gwangsu Kim, Matt Olfat, Mark Hasegawa-Johnson, and Chang D. Yoo.

If you plan to use this repository or cite our preprint, please use the following bibtex format:

```latex
@inproceedings{lee2022fairpca,
	title={{Fast and Efficient MMD-Based Fair PCA via Optimization over Stiefel Manifold}},
	volume={36},
	url={https://arxiv.org/abs/2109.11196.pdf},
	number={7},
	booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
	author={Lee, Junghyun and Kim, Gwangsu and Olfat, Mahbod and Hasegawa-Johnson, Mark and Yoo, Chang D.},
	year={2022},
	month={Jun.},
	pages={7363-7371}
}

```

Most of the codes (mbfpca.m, mmd.m, fairness_metric.m) are written in MATLAB.

We used the Python implementation of FPCA, as provided by (Olfat & Aswani, AAAI'19).

## Requirements

### ROPTLIB: Riemannian Manifold Optimization Library

- Windows
- MacOS ([2020-08-12 release](https://github.com/whuang08/ROPTLIB/releases/tag/0.8))
- Ubuntu

### Python packages

These are need *only* when FPCA is to be run.

- MOSEK
- 

## Reproducing Experiments

First, run the following three commands (in this order) in MATLAB command line:

```matlab
mex -setup
GenerateMyMex;
MyMex DriverMexProb;
```

All the results and plots are available, but if one wants to recreate the results from the bottom, refer to the following subsections.

(Also, some filenames may contain 'stfpca' or 'STFPCA'; this is an artifact of previous version. One can simply fix it to 'MBFPCA' and run the codes, if there's any related error.)

### Section 8.1 Synthetic data \#1

Follow these steps:

1. Go to *synthetic1* folder
2. Run **synthetic1_generate.m** for generating the synthetic dataset
3. Run **synthetic1.m** for generating scatter plots of (vanilla) PCA and MbF-PCA
   - Figure 2 is for MbF-PCA
   - Figure 3 is for (vanilla) PCA
4. To run FPCA by (Olfat & Aswani, AAAI'19), go to *FPCA* folder and run **runSynthetic1()** in **problem.py**
5. Then come back to *synthetic1* folder and run **synthetic1_plot_fpca.m** to generate scatter plot of FPCA

### Section 8.2 Synthetic data \# 2

Follow these steps:

1. Go to *synthetic2* folder
2. Run **synthetic2_generate.m** for generating the synthetic dataset
3. Run **synthetic2.m** for running/storing results of (vanilla) PCA and MbF-PCA
   (Results := mmd_train, exp_vars_train, mmd_test, exp_vars_test, runtimes)
4. To run FPCA by (Olfat & Aswani, AAAI'19), go to *FPCA* folder and run **runSynthetic2()** in **problem.py**
5. Then come back to *synthetic2* folder and run **synthetic2_analysis_fpca.m** to store results of FPCA
6. Run **boxplot.R** for generating the boxplots



cf. *mbfpca_1, mbfpca_2, ...* corresponds to results of MbF-PCA with different value of tau; see line 12 of **synthetic2.m**

### Section 8.3 UCI datasets

Follow these steps:

1. Run preprocessing python files for each dataset in the folder *datasets* folder
2. Go to *uci* folder and run **uci.m** for running/storing loading matrices of (vanilla) PCA and MbF-PCA
3. To run FPCA by (Olfat & Aswani, AAAI'19), go to *FPCA* folder and run **runUCI()** in **problem.py**
4. Then come back to *uci* folder and run **uci_pca_analysis.m, uci_fpca_analysis.m, uci_mbfpca_analysis.m** to store results of PCA, FPCA, MbF-PCA, respectively.
5. Check the resulting .csv files in the folders *pca, fpca, mbfpca*



cf. Watch out for the target dimension d. If one wants, he/she can change d to some other value.

ccf. It is recommended that the loading matrices are stored in separate folder as they will get overwritten if the same command with different hyper parameters is run...

## Usages

### mbfpca.m

There are 6 required inputs to the function:

#### X

The data matrix, which should be of the dimension n x (p + 2) where n is the number of samples, p is the original dimension of the data (i.e. number of features).

Last column should be the **sensitive** group label (in binary {0, 1}) while the second to last column should be the **downstream** classification label (in binary {0, 1})

#### d

The target dimension, which should satisfy d < p

#### fairness

Type of fairness to be pursued: **DP** (Demographic Parity), **EOP** (Equal Opportunity), **EOD** (Equalized Odds)

#### sigma

The bandwidth of the MMD for the whole optimization process. Usually this is set using the median-heuristic applied to the dimensionality-reduced data matrix via PCA; see, for instance, line 53-58 of test_mbfpca.m

#### rho0

Starting value of rho. This is usually set as 0.1/start_mmd, where start_mmd is the MMD of the initial distribution; see, for instance, line 61-64 of test_mbfpca.m

#### tau

Tolerance level for fairness. Usually set to values among {1e-3, 1e-4, 1e-5, 1e-6}.
