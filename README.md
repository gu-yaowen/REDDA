# REDDA
![Visits Badge](https://img.shields.io/badge/dynamic/json?label=visits&query=$.message&color=blue&url=https://hits.dwyl.com/gu-yaowen/REDDA.json)

Code and Dataset for "REDDA: integrating multiple biological relations to heterogeneous graph neural network for drug-disease association prediction".
# Reference
If you make advantage of the REDDA model or use the datasets released in our paper, please cite the following in your manuscript:

```
@article{10.1016/j.compbiomed.2022.106127,
author = {Yaowen Gu, Si Zheng, Qijin Yin, Rui Jiang, Jiao Li},
title = "{REDDA: Integrating multiple biological relations to heterogeneous graph neural network for drug-disease association prediction}",
journal = {Computers in Biology and Medicine},
year = {2022},
month = {11},
issn = {0010-4825},
doi = {10.1016/j.compbiomed.2022.106127},
}
```
# Benchmark Dataset
Our proposed drug repositioning benchmark dataset (**Kdataset**), including **894** drugs, **454** diseases, **18,877** proteins, **20,561** genes, **314** pathways, **2,704** drug-disease associations, **4,397** drug-protein associations, **18,545** protein-gene associations, **25,995** gene-pathway associations, **19,530** pathway-disease associations, **201,382** protein-protein interactions, **712,546** gene-gene interactions, and **1,669** pathway-pathway interactions. The files are as shown:
> ``Omics`` \
The node mappings of benchmark identifiers and external identifiers.
>> * ``drug.csv`` \
  Benchmark IDs -- DrugBank IDs -- SMILES strings
>> * ``protein.csv`` \
  Benchmark IDs -- UniProt IDs -- Amino acid sequences
>> * ``gene.csv`` \
Benchmark IDs -- Entrez IDs
>> * ``pathway.csv`` \
Benchmark IDs -- KEGG IDs
>> * ``disease.csv`` \
Benchmark IDs -- MeSH IDs

> ``Interactions`` \
The edges whose start nodes and destination nodes belong to the same node type.
>> * ``drug-drug.csv`` \
Drug1 IDs -- Drug2 IDs -- ECFP4 similarity
>> * ``protein-protein.csv`` \
Protein1 IDs -- Protein2 IDs -- Combined score (extracted from STRING)
>> * ``gene-gene.csv`` \
Gene1 IDs -- Gene2 IDs
>> * ``pathway-pathway.csv`` \
Pathway1 IDs -- Pathway2 IDs
>> * ``disease-disease.csv`` \
Disease1 IDs -- Disease2 IDs -- MeSH similarity

> ``Associations`` \
The edges whose start nodes and destination nodes belong to different node types.
>> * ``drug-protein.csv`` \
Drug IDs -- Protein IDs
>> * ``protein-gene.csv`` \
Protein IDs -- Gene IDs
>> * ``gene-pathway.csv`` \
Gene IDs -- Pathway IDs
>> * ``pathway-disease.csv`` \
Pathway IDs -- Disease IDs
>> * ``Kdataset.csv`` \
Drug IDs -- Disease IDs

Other files:
* ``drug_drug_baseline.csv``: binarized drug-drug matrix with a demension of **894×894**. Note that the binary values are calculated by a Top15 filtering of drug-drug similarity.

* ``disease_disease_baseline.csv``: binarized disease-disease matrix with a demension of **454×454**. Note that the binary values are calculated by a Top15 filtering of disease-disease similarity.

* ``Kdataset_baseline.csv``: binarized drug-disease matirx with a demension of **894×454**.

Similarly, a re-curated **B-dataset** is also used and stored in this repo, with the same file naming and hierarchy division, including **269** drugs, **598** diseases, **6,040** proteins, **18,416** drug-disease associations, **2,107** drug-protein associations, **17,631** protein-disease associations, and **592,926** protein-protein interactions.

# REDDA model
![REDDA architecture](https://github.com/gu-yaowen/REDDA/blob/main/model_structure.png)
## Requirement
Pytorch >= 1.7.0

DGL >= 0.5.2

## Run
    python main.py -id {DEVICE ID} -da Kdataset_baseline -sp {SAVED PATH}
    Optional Argument:
      -fo Number of k-folds cross-validation
      -ep Number of epoches
      -lr Learning rate
      -wd Weight decay
      -pa Patience in early stopping
      -hf Dimension of hiddent feats
      -he Number of heads in graph attention
      -dp Dropout rate
 
# Contact
We welcome you to contact us (email: gu.yaowen@imicams.ac.cn) for any questions and cooperations.
