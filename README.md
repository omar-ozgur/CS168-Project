# Brain Lesion Segmentation from FLAIR MRIs

## Introduction

Automatic brain lesion segmentation can be extremely crucial in helping doctors quickly and effectively identify potentially life threatening conditions and symptoms that cause the brain lesions.

We chose to focus on leveraging two key techniques in automatic brain lesion segmentation:
1. Image processing via thresholding with k-means clustering
2. Unsupervised machine learning using the pix2pix conditional generative adversarial network

While we initially intended on running our models on T1w, T2w and FLAIR MRIs, we chose to focus only on FLAIR MRIs due to our limited dataset that was procured from the Neuroscience department at the University of California, Los Angeles. Before beginning the project, we reviewed some of the state of the art methods for brain lesion segmentation which can be found [here](https://drive.google.com/file/d/16AWvtvrqFEP6ZTUXRjxh-rrSpnvnF8ie/view?usp=sharing).

![Screenshot](images/comparison.png)

## Directory Structure
```
.
├── data
|   ├── 1
|   |   ├── Features
|   |   |   ├── *.dcm
|   |   ├── Labels
|   |   |   ├── *.dcm
|   ├── 2
|   ...
|   └── 23
├── output
|   ├── pix2pix/lesions_test
|   ├── thresholding
|   |   ├── high
|   |   |   ├── 1
|   |   |   |   ├── Original.jpg
|   |   |   |   ├── ROI.jpg
|   |   |   |   ├── Segmented.jpg
|   |   |   ...
|   |   |   └── 23
|   |   ├── medium
|   |   ├── low
├── thresholding
|   ├── main.py
|   └── requirements.txt
├── images
|   ├── comparison.png
|   └── thresholding.png
├── tools
|   ├── mri_to_png.py
|   ├── process.py
|   ├── split.py
|   ├── test.py
|   └── tfimage.py
├── pix2pix.py
├── Makefile
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Run Instructions

This project is broken into two independent pipelines, with one for the image processing and the other for the machine learning. To begin, clone the git repository:
```bash
git clone git@github.com:omar-ozgur/CS168-Project.git
```

### Image Processing

1. Run `cd thresholding` to move to the thresholding directory.
2. To setup thresholding dependencies, run `make setup`.
3. To run the thresholding algorithm, run `make run`.

This process will run on data in the root "data/thresholding" directory, and output results to the root "output/thresholding" directory.

### Machine Learning

#### Pre-requisites

* Tensorflow 1.4.1 or above

#### Recommended

* Tensorflow GPU + cuDNN

The pix2pix pipeline is custom built with the help of the scripts in the `tools` directory. The originals are read from the `data/*/Features` directory and the targets are read from the `data/*/Labels` directory. The pipeline runs as follows:
1. Get all required dependencies for running the pix2pix pipeline. This can be run via `make setup`
2. Clean all prior data from `output/pix2pix/`. This can be run via `make clean`
3. Recursively convert dicom images from `data` folder to _pngs_ into `output/pix2pix/inputs` and `output/pix2pix/outputs`. This can be run via `make convert`
4. Resize all pngs to 256x256 images into `output/pix2pix/in_resize` and `output/pix2pix/out_resize`. This can be run via `make resize`
5. Combine the inputs and desired outputs into a single side by side png for pix2pix in the `output/pix2pix/combined` directory. This can be run via `make combine`
6. Segregate the combined data into _training_ and _testing_ sets in `output/pix2pix/combined/train` and `output/pix2pix/combined/val` respectively. This can be run via `make split`

**Note that all of this can be completed by running the following cmake command:**
```bash
make run
```
7. Train the model by running `make train` which will store training data into `output/pix2pix/lesions_train`
8. Test the model on validation data by running `make test` which will create a HTML file of comparison in `output/pix2pix/lesions_test`

## Results

When reviewing the results of the segmentation, we chose to focus on metrics of **sensitivity** and **specificity** to denote how well our models could detect brain lesions in the presence of actual brain lesions, and how well it could ignore false segmentation when no brain lesion existed.

The pix2pix library was trained on 480 MRI slices from 23 patients, and tested against 118 MRI slices from the same patients. It was run with Tensorflow GPU on a NVIDIA GTX 1070 with an average training time of 223 minutes.

We found that the pix2pix model was most successful at accurately segmenting the brain lesions primarily due to the fact that the generator and discriminator within the pix2pix cGAN trained on a varied data set. More details of the results and their significance can be found in our [report](https://docs.google.com/document/d/1RftqDMoXXs4qWlg8siRhz6chSB2Nw78hr0g1FRWQ9Zw/edit?usp=sharing).

## Troubleshooting

Note that due to a large dataset, the data folder _might not be present_. In this case, you can download our dataset from [this link](https://drive.google.com/file/d/1GwU1cTqgz_Tw-MLOeuwevG2DPiAjQNOE/view) and unzip it manually into a `data` folder, or you can do this conveniently with `make download` while in the root directory. This will automatically download the data, unzip into `data` and remove the zip file.

## Credit

The pix2pix library is based on a very influential [paper](https://arxiv.org/pdf/1611.07004v1.pdf) of Image to Image translation by Phillip Isola et al.

We used the Tensorflow implementation as created by [affinelayer](https://github.com/affinelayer/), utilizing only the necessary components within our own pipeline. [Source](https://github.com/affinelayer/pix2pix-tensorflow)

The image processing approach that utilizes thresholding was inspired by a similar approach documented by Howard Chen. [Source](https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/)
