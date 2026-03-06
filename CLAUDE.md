# What 

This is meant to be an interpretable spin on the succesful MPNN based GNN architecture in cheminformatics. Models such as chemprop or MiniMol have proven to be very succesful, probably because graphs are natural (if incomplete) representations of molecules. A major downside is that these models are generally poorly interpretable which is particularly key when used in structure-activity relationship (SAR) landscapes, where interpretable predictions could guide a chemists decision as to what molecules to test next. 

## Interpretability Approach: Integrated Gradients

Integrated gradients (Sundararajan et al., ICML, 2023) is a method for attributing a deep neural network’s prediction to its input features. The core idea is to integrate the gradients of the output taken along a linear path from a baseline input to the input at hand, see Eq. (3). Mathematically, for a neural network F(x), an input x and baseline input  (e.g. the zero input), the attribution for the ith feature is:

{{\rm{IntegratedGrads}}}_{i}(x)=\left({x}_{i}-{x}_{i}^{{\prime} }\right)\times \mathop{\int}\nolimits_{0}^{1}\frac{\partial F\left({x}^{{\prime} }+\alpha \times \left(x-{x}^{{\prime} }\right)\right)}{\partial {x}_{i}}\,{\rm{d}}\alpha .

The method satisfies important axioms like sensitivity (if inputs differ in one feature but have different predictions, that feature should receive attribution) and implementation invariance (attributions are identical for functionally equivalent networks). The method is readily available for Hugging Face Transformer models through the transformers-interpret package (https://github.com/cdpierse/transformers-interpret). 

# Why

The long-term (ideal version) goal of this model would be a pre-trained, transformer-on-graph architecture that can be fine-tuned on biological activity data (namely activity as a drug, toxicity, etc.). Using an integrated gradients approach to interpretability (see above) we can see which functional groups (or atoms, scaffolds, etc) are helpful or detrimental to the drugs performance. Based on this we could more efficiently exploit the SAR landscape. The ideal interface would be a Chemdraw style editor where a chemist could come up with new structures and immediately see whether the score(s) improved or not and which atoms positively/negatively contribute to it.

# How

* Keep the codebase as simple as possible. Use as many standard, well-maintained, common packages (transformers, pytorch, scipy, etc) as possible. 
* Always keep in mind how these models get deployed in the end: They will pre-train on very large, supervised datasets (including hyperparameter optimization) which have multiple labels. Multi-task learning would be ideal where we train end-to-end including the prediction heads (a separate one for each task), these get deleted after pre-training and we only keep the shared parameters lower down. In general, we will use the same datasets as the pre-trained GNN MiniMol and wherever unclear we will make the same architectural decisions (https://github.com/graphcore-research/minimol, https://arxiv.org/pdf/2404.14986).
* Then the models get pre-trained on some (presumably smaller) dataset and is then tasked with outputting a binary prediction (between 0 and 1) as to how active a given molecule will be. 
* Ultimately the model should be light-weight, fast, and easy to interface. 
* Assume that all molecules will be inputted as SMILES. For now, we don't need to support other formats. Do check whether a SMILES is valid and we will have to do graph construction as well. 
* Make sure there is way to preconstruct all graphs of a dataset beforehand. Otherwise, each time we will have to reconstruct the same graphs from the SMILES of the molecules even though we could have reused this. Essentially like a pre-tokenized dataset. 