# Setup
** requirements 
- Ubuntu 18.04
- Python >=3.5
- Matlab

**Python packages
- sklearn 0.0
- numpy 1.18.3
- scipy 1.4.1
- grakel 

# Download MDBD dataset from the link below
https://drive.google.com/drive/folders/1LCNmY648XAbskm7ewzqLfpr6-VhF6BE3?usp=sharing

# Run
** Generate graph data **
```
python3 shrec_generate_dataset.py
```
The output is stored in "synthetic_adj_nor.txt" file.

** Shape retrieval **
```
python3 shrec_shape_retrieval.py
```
The result is stored in "synthetic_T1_whole.txt" file.

** Evaluation **
Open Matlab, run
```
results = SHREC14Eval('synthetic_T1_whole.txt', 'synthetic_test.cla', 'outStats.txt', 'outPR.txt', 1)
