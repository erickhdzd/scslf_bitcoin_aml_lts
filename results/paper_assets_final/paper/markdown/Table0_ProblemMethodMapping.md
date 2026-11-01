| Challenge | Consequence | Methodological response |
|---|---|---|
| Class imbalance | Overall accuracy can hide poor performance on the illicit class. | Imbalance-aware metrics and optimization of the decision threshold. |
| Complex dependencies among attributes | Discriminative patterns may depend on relationships nonlinear interactions among multiple features. | Transformer to model global interactions among transaction features. |
| Structural relationships between transactions | Observations are not independent and their neighborhood may contain relevant information. | GraphSAGE to learn representations from topological context. |
| Complementary information across models | A single representation may omit signals present in other views of the data. | Late Fusion to combine probabilities produced by complementary models. |
| Experimental variability | A single run may produce conclusions that depend on random initialization. | Evaluation with multiple seeds, means, standard deviations and paired statistical tests. |
| Need for verifiable results | Results may be difficult to audit or reproduce independently. | Deterministic asset pipeline, manifest, hashes SHA-256 and reproducibility metadata. |
