# ImgX Datasets

A [TFDS](https://www.tensorflow.org/datasets/add_dataset)-based python package for data set
building.

Current supported data sets are listed below. Use the following commands to (re)build all data sets.

```bash
make build_dataset
```

## Male pelvic MR

### Description

This data set from [Li et al. 2022](https://zenodo.org/record/7013610#.Y1U95-zMKrM) contains 589
T2-weighted labeled images which are split for training, validation and testing respectively.

### Download and Build

Use the following commands at the root of this repository (i.e. under `ImgX/`) to automatically
download and build the data set, which will be built under `~/tensorflow_datasets` folder.
Optionally, add flag `--overwrite` to rebuild/overwrite the data set.

```bash
tfds build imgx_datasets/male_pelvic_mr
```

## AMOS CT

### Description

This data set from [Ji et al. 2022](https://zenodo.org/record/7155725#.ZAN4BuzP2rO) contains 500 CT
labeled images which has been split into 200, 100, and 200 images for training, validation, and test
sets. But test set labels were not released, therefore validation is further split into 10 and 90
images for validation and test sets.

### Download and Build

Use the following commands at the root of this repository (i.e. under `ImgX/`) to automatically
download and build the data set, which will be built under `~/tensorflow_datasets` folder.
Optionally, add flag `--overwrite` to rebuild/overwrite the data set.

```bash
tfds build imgx_datasets/amos_ct
```

## Muscle Ultrasound

### Description

This data set from [Marzola et al. 2021](https://data.mendeley.com/datasets/3jykz7wz8d/1) contains
3910 labeled images, which has been split into 2531, 666, and 713 images for training, validation,
and test sets.

### Download and Build

Use the following commands at the root of this repository (i.e. under `ImgX/`) to automatically
download and build the data set, which will be built under `~/tensorflow_datasets` folder.
Optionally, add flag `--overwrite` to rebuild/overwrite the data set.

```bash
tfds build imgx_datasets/muscle_us
```

## Brain MR

### Description

This data set from [Baid et al. 2021](https://arxiv.org/abs/2107.02314) contains 1251 labeled images
which are split for training, validation and testing respectively.

### Download and Build

#### Manual Download

This data set requires manual data downloading from
[Kaggle](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1). using
[kaggle API](https://www.kaggle.com/docs/api). The
[authentication token](https://www.kaggle.com/docs/api#getting-started-installation-&-authentication)
shall be obtained and stored under `~/.kaggle/kaggle.json`.

Then, execute the following commands to download and unzip files. Afterward, return to `ImgX/`
folder (`/app/ImgX` for docker).

```bash
mkdir -p ~/tensorflow_datasets/downloads/manual/BraTS2021_Kaggle/BraTS2021_Training_Data/
cd ~/tensorflow_datasets/downloads/manual/BraTS2021_Kaggle/BraTS2021_Training_Data/
kaggle datasets download -d dschettler8845/brats-2021-task1
unzip brats-2021-task1.zip
tar xf BraTS2021_Training_Data.tar
rm BraTS2021_00495.tar
rm BraTS2021_00621.tar
rm BraTS2021_Training_Data.tar
rm brats-2021-task1.zip
```

This way under `BraTS2021_Kaggle/` exist folders per sample. For example, files corresponding to uid
`BraTS2021_01666` should be located at
`~/tensorflow_datasets/downloads/manual/BraTS2021_Kaggle/BraTS2021_Training_Data/BraTS2021_01666/`
under which there are five files:

- `BraTS2021_01666_flair.nii.gz`,
- `BraTS2021_01666_t1.nii.gz`,
- `BraTS2021_01666_t1ce.nii.gz`,
- `BraTS2021_01666_t2.nii.gz`,
- `BraTS2021_01666_seg.nii.gz`.

#### Automatic Build

Use the following commands at the root of this repository (i.e. under `ImgX/`) to automatically
build the data set, which will be built under `~/tensorflow_datasets` folder. Optionally, add flag
`--overwrite` to rebuild/overwrite the data set.

```bash
tfds build imgx_datasets/brats2021_mr
```
