# Data Preprocessing

## FiveK

**Download dataset.** Please download the FiveK dataset from https://data.csail.mit.edu/graphics/fivek/ to `data/fivek_dataset`
```bash
python -m datasets.fivek_process
```
**Subsample Datasets.** Create datasets with limited number of training examples.
```bash
python -m datasets.fivek_subsample --dataset_path=data/fivek_dataset_processed/fivek_patches_3 --seed=0 --max_images=100
```

## NOD

**Download the dataset.** Please download the NOD dataset from the official website to `data/NOD`: https://github.com/igor-morawski/RAW-NOD

**Download annotations**
```bash
git clone https://github.com/igor-morawski/RAW-NOD data/RAW-NOD
mv data/RAW-NOD/annotations/Sony/raw_str_labeled_new_Sony_RX100m7_test.json data/RAW-NOD/annotations/Sony/raw_new_Sony_RX100m7_test.json 
```
**Preprocess data**
```bash
python -m datasets.nod_process
python -m datasets.nod_process_od
```
**Subsample Datasets.** Create datasets with limited number of training examples.
```bash
python -m datasets.nod_subsample --dataset_path=data/NOD_processed/NOD_patches_3 --seed=0 --max_images=25
python -m datasets.nod_subsample --dataset_path=data/NOD_processed/NOD_h416_d32 --seed=0 --max_images=100
```

## Cityscapes

Download the dataset from https://www.cityscapes-dataset.com to `data/Cityscapes`
```bash
python -m datasets.cityscapes_process
```

## BDD100K

Download the dataset from http://bdd-data.berkeley.edu to `data/BDD`
```bash
python -m datasets.bdd_process
```

