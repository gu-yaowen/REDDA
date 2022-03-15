# MODDA
Code and Dataset for "MODDA: a drug repositioning method based on a large-scale multi-omics heterogeneous network".
# Reference
If you make advantage of the MODDA model or use the datasets released in our paper, please cite the following in your manuscript:

TBD

# Benchmark Dataset
Our proposed multi-omics-based drug repositioning benchmark dataset includes **894** drugs, **454** diseases, and **2704** drug-disease associations. The files are as shown:
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
The edges whose start nodes and destination nodes belong to the same omics type.
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
The edges whose start nodes and destination nodes belong to different omics types.
>> * ``drug-protein.csv`` \
Drug IDs -- Protein IDs
>> * ``protein-gene.csv`` \
Protein IDs -- Gene IDs
>> * ``gene-pathway.csv`` \
Gene IDs -- Pathway IDs
>> * ``pathway-disease.csv`` \
Pathway IDs -- Disease IDs
>> * ``KFCdataset.csv`` \
Drug IDs -- Disease IDs

Other files:
* ``drug_drug_baseline.csv``: binarized drug-drug matrix with a demension of **894×894**. Note that the binary values are calculated by a Top15 filtering of drug-drug similarity.

* ``disease_disease_baseline.csv``: binarized disease-disease matrix with a demension of **454×454**. Note that the binary values are calculated by a Top15 filtering of disease-disease similarity.

* ``KFCdataset_baseline.csv``: binarized drug-disease matirx with a demension of **894×454**.

# MODDA model
![MODDA architecture](https://github.com/gu-yaowen/MODDA/blob/main/model_structure.png)
## Requirement
Pytorch == 1.7.0

DGL == 0.5.2
## Run
    python main.py -id {DEVICE ID} -da KFCdataset_baseline -sp {SAVED PATH}
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
