# Machine learnig to segment neutron images

|Status | Downloads |
|:------:|:----------:|
|[![Build Status](https://www.travis-ci.com/ImagingLectures/MLSegmentation4NI.svg?branch=main)](https://www.travis-ci.com/ImagingLectures/MLSegmentation4NI)|[<img src="downloadbook.svg" height="50px"/>](MLSegmentation4NI.pdf)  &nbsp;&nbsp;&nbsp; [<img src="https://upload.wikimedia.org/wikipedia/commons/3/38/Jupyter_logo.svg" height="50px"/>](https://github.com/ImagingLectures/MLSegmentation4NI/tree/gh-pages/lecture/ML4NeutronImageSegmentation.ipynb) &nbsp;&nbsp;&nbsp; [<img src="np_presentation.svg" height="50px"/>](https://imaginglectures.github.io/MLSegmentation4NI/ML4NeutronImageSegmentation.slides.html)|
## Getting started
If you want to run the notebook on your own computer, you'll need to perform the following step:
- You will need to install Anaconda
- Clone the lecture repository (in the location you'd like to have it)
```bash
git clone https://github.com/ImagingLectures/MLSegmentation4NI.git
```
- Enter the folder 'MLSegmentation'
- Create an environment for the notebook
```bash
conda env create -f environment. yml -n MLSeg4NI
```
- Enter the environment
```bash 
conda env activate MLSeg4NI
```

- Start jupyter and open the notebook ```lecture/ML4NeutronImageSegmentation.ipynb```

- Use the notebook

- Leave the environment
```bash
conda env deactivate
```


## Lecture outline

### Introduction
-	Introduction to neutron imaging
  - Some words about the method
  - Contrasts
- Introduction to segmentation
  - What is segmentation
  - Noise and SNR
- Problematic segmentation tasks
  - Intro
  - Segmenation problems in neutron imaging

### Limited data problem
-	Training data from NI is limited
-	Augmentation
-	Transfer learning

### Unsupervised segmentation
-	e.g. k-means

### Supervised segmentation
-	e.g. k-NN, decision trees
-	NNs for segmentation

### Final problem: Segmenting root networks in the rhizosphere using convolutional NNs
-	Problem definition
-	NN model
-	Loss functions
-	Training
-	Results

### Future Machine learning challenges in NI
