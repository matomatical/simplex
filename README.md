Simplex replication
===================

Stage 1: Belief state geometry
------------------------------

Replicate Shai et al. (2024) "Transformers represent belief state geometry in
their residual stream" ([arXiv](https://arxiv.org/abs/2405.15943)).

* [x] Implement HMM engine and Z1R + Mess3 HMMs
* [x] Implement transformer architecture
  * [ ] A single head with a smaller head dimension
* [ ] Train for 1m steps
* [ ] Implement ground truth mixed state presentation engine
* [ ] Find linear subspace
* [ ] Make the famous plot

Stage 2: Factorisation
----------------------

Replicate Shai et al. (2026) "Transformers learn factored representations"
([arXiv](https://arxiv.org/abs/2602.02385)).

* [ ] Implement (approximately) factorised HMM
* [ ] Match their transformer architecture
* [ ] Train transformer
* [ ] Recover factorised state representation
* [ ] Replicate figure 4b

Stage 3: Abstraction
--------------------

Extend these by studying approximate bisimulation.

* [ ] Implement (approximately) bisimulated HMM.
* [ ] Train a transformer on it
* [ ] Recover simplices
* [ ] Identify the point at which the model distinguishes states
* [ ] Repeat with larger transition systems.

