# README.md
[![DOI](https://zenodo.org/badge/833483470.svg)](https://zenodo.org/doi/10.5281/zenodo.12958255)

## Installation instructions

The code is written in Python 3 (>= 3.9). We recommend to create a conda environments and install the dependencies in `requirements.txt`. We used pytorch 1.12.1 with cuda 11.3. For futher installation details about pytorch please see https://pytorch.org/get-started/previous-versions. The expected installation time on a computer without GPU is 30 to 60 minutes.

## Data preparation for training

1. Download PubChem dataset from [https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF](https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF)  and place all the *.gz and *.gz.md5 files in `data/sdf` . As an example two gz files are already placed in the folder

2. The sdf files needs to be preprocessed to generate a file which contains standardized smiles (see `lib/dataset/chem.py` for the function) and pubchem id. Along with those informations we also retain the original pubchem smiles. The preprocessing can be run as follows. We recommend to use >100 CPUs or a SLURM platform for processing the whole dataset. The script will generate a pickle file named `pubchem.pkl`.
   
```bash
$ export ncpus=1
$ python prepare_dataset.py data/sdf data $ncpus
```

3. We compute the ecfp4 without counts with 1024 bits as
   
```bash
$ mkdir data/ecfp4
$ export ncpus=1
$ python compute_ecfp4.py data/pubchem.pkl data/ecfp4 $ncpus
```

4. We compute all the pairwise tanimoto similarities. It takes about 10 days on 16 GPUs for the whole dataset. The similarities are stored as binary files where only pubchem id of the compounds are saved with their corresponding tanimoto similarities. This compressed form takes approximately 1 TB of HD space.
   
```bash
$ export gpu_id=0
$ export ngpus=1
$ python compute_pairwise_ecfp4.py data/ecfp4 data/ecfp4/dat $gpu_id $ngpus
```

5. Finally, we need to compute some cache data to make the training more efficient as
   
```bash
$ mkdir -p cache/ecfp4
$ python train_preprocessing.py config/ranking_loss_ecpf4.yml
```

6. Starting from ecfp4 without counts it is possible to generate ecfp4 with counts as
   
```bash
$ ncpus=8
$ python compute_pairwise_ecfp4_with_counts.py data/pubchem.pkl data/ecfp4/dat data/ecfp4_wc/dat $ncpus
```

## Trainining

The training script `train.py` supports multiple GPUs. You can run it as described in the example below. It takes as input a yaml config file, the output folder where checkpoints and other metadata are save, and the number of gpus.

```bash
$ export ngpus=1
$ python train.py config/ranking_loss_ecpf4.yml results/ecfp4 $ngpus
```

## Generate compounds (DEMO)

To generate compounds just run the following script. All the pretrained models used in the paper are inside the folder `paper_checkpoints`

```bash
$ python generate_smiles.py \
             --model paper_checkpoints/ecfp4_with_counts_with_rank \
             --input-smiles examples/smiles.txt \
             --samples 1000 \
             --result-path samples.csv
```

The script will generate 1,000 compounds collected in csv. The csv contains 5 columns: input_smiles, generated_smiles, nll, tanimoto, is_valid. The expected run time on a computer without GPU is 3 to 5 minutes for each input smiles.

## Quantitative results of the paper
The script `make_table_1.py` can be used to reproduce Table 1. Each row will take 90 to 120 minutes on GPU to be computed
```bash
$ python make_table_1.py paper_checkpoints/ecfp4_no_counts_no_rank/config.yml paper_checkpoints/ecfp4_no_counts_no_rank/weights.ckpt paper_checkpoints/vocabulary.pkl
$ python make_table_1.py paper_checkpoints/ecfp4_no_counts_with_rank/config.yml paper_checkpoints/ecfp4_no_counts_with_rank/weights.ckpt paper_checkpoints/vocabulary.pkl
$ python make_table_1.py paper_checkpoints/ecfp4_with_counts_no_rank/config.yml paper_checkpoints/ecfp4_with_counts_no_rank/weights.ckpt paper_checkpoints/vocabulary.pkl
$ python make_table_1.py paper_checkpoints/ecfp4_with_counts_with_rank/config.yml paper_checkpoints/ecfp4_with_counts_with_rank/weights.ckpt paper_checkpoints/vocabulary.pkl
```

Furthermore, with small modifications the script `make_table_1.py` can be used to reproduce all the othjer quantitative results. The test data (ChEMBL series and TTD) is located in `test_data`.


Test data: CHEMBL/TTD
