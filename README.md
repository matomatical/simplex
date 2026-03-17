Simplex replication
===================

Stage 1: Belief state geometry
------------------------------

Replicate Shai et al. (2024) "Transformers represent belief state geometry in
their residual stream" ([arXiv](https://arxiv.org/abs/2405.15943)).

* [x] Implement HMM engine and Z1R + Mess3 HMMs
* [x] Implement transformer architecture
  * [x] Allow a single head with a smaller head dimension
* [x] Train for 1m steps
  * [x] Check what loss we expect
* [x] Implement ground truth mixed state presentation engine
* [x] Find linear subspace (linear probe on residual stream)
* [x] Make the famous plot (live training visualisation)

Stage 2: Factorisation
----------------------

Replicate Shai et al. (2026) "Transformers learn factored representations"
([arXiv](https://arxiv.org/abs/2602.02385)).

* [x] Implement factorised HMM and visualisation
* [x] Match their transformer architecture
* [x] Simple tetrahedron case
  * [x] Train transformer
  * [x] Recover (factorised) state representation (probes)
  * [x] Epsilon sweep: turns out joint probe beats factored at all noise levels
* [x] Scale to 5-factor model
  * [x] Train transformer
  * [x] Recover (factorised) state representations (probes, PCA)
  * [x] Epsilon sweep
* [ ] Replicate figure 4b, stagewise development?

The batch sizes these guys are using are huge! Training will take like 6 days
on a TPU v4-8...

Stage 3: Non-egodicity
----------------------

For MATS work trial they suggested I consider non-ergodic processes that are
formed from unioning several Mess3 processes together, with a decision at the
beginning that decides which to enter and then for the rest of the sequence we
stay there.

Thinking phase:

* [x] Formalise the process
* [x] Derive the optimal predictor's belief distribution
* [x] Predict possible geometries (based on intuition and math)

Implementation phase:

* [ ] Construct the training data set
* [ ] Training loop to train a transformer on this data
* [ ] Probes for the geometries, in comparison to individual process geometries

Report phase

* [ ] Initialise report
* [ ] Why is this setting interesting
* [ ] Prediction
* [ ] Report experimental results

Extension phase

* [ ] Plan an extension, for example around continuous variables and the phase
  transition to those.



Stage 4: Abstraction
--------------------

Extend these by studying approximate bisimulation.

* [ ] Implement (approximately) bisimulated HMM.
* [ ] Train a transformer on it
* [ ] Recover simplices
* [ ] Identify the point at which the model distinguishes states
* [ ] Repeat with larger transition systems.

