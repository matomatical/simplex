Please implement the following experiment, and send us a PDF writeup of the
results, alongside the code. It may be helpful to refer to our publications for
definitions of processes and how to derive belief state geometry, in
particular, Transformers Learn Factored Representations and the references
therein. For computational feasibility, we recommend making the neural networks
small (e.g., 1–3 layers, context window 8–16).

1. Construct a non-ergodic training dataset, consisting of Mess3 processes with
   different parameters, where each training sequence is generated entirely by
   one Mess3 ergodic component. Train a transformer on this data via next-token
   prediction. Why do you think this type of structure is interesting and/or
   relevant to language models?

2. Make a pre-registered prediction (honor code) as to what geometry the
   activations should take, and how it should change with context position and
   across layers. Derive this prediction mathematically as far as you can, and
   separately give your intuition for what the geometry should look like. You
   will not be penalized for getting this wrong—we are interested in both your
   formal reasoning and your geometric intuition. Are there multiple possible
   geometries you can think of?

3. After training, analyze the residual stream geometry. What structure is
   there? How does it relate to the belief geometries of the component
   processes?

4. Perform at least one additional analysis of your choosing that you think is
   interesting or informative. Tell us why you chose it. If you don’t have time
   to implement it, describe what you would do.

