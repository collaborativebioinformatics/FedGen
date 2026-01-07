# Fed_learning_infrastructure
Federated Learning (FL) Infrastructure &amp; Synthetic Data


## Topics

### 1. Synthetic data generation
- Genotypes
    - LD structure
    - Maybe sample from 1000 genomes?
    - Look at genotype simulators from the original gdoc
- Phenotype (0/1) - e.g., presence of Parkinson's disease
    - Take into account that if >1 phenotype, there is no covariance between phenotypes
- Assume that all have the same build (hg38)
- Generate in PLINK format (bed/bim/fam)
- Simulate per-site variability for genotypes and phenotypes

### 2. Federated learning infrastructure
- Server, client, admin specs (project.yml, NVFlare Dashboard on AWS)
- Set up the **FL server** on AWS
- ~10 Brev instances/**clients**
- Imbalance between sites of data distribution

### 3. Learning task
- Logistic regression model (sklearn-based)
- PyTorch model
- PLINK format


## Flow Chart

![flow-chart](./resources/Fed_learning_infrastructure.drawio.svg)
