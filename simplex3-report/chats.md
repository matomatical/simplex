❯ OK, let's leave it running. I am a bit confused about what to predict for
representations. One prediction is that the transformer decomposes its
representations into the two component belief processes and the meta belief process.
 However, this has the same number of dimensions as if the model just represents the
 full joint belief process. Is this decomposition actually predicting anything at a
representational level? Or only a mechanistic level? Ultrathink.

● This is a great question. Let me think through it carefully.

  The dimensionality argument: The joint belief lives on a 5-simplex (6 states, sum
  to 1 → 5 degrees of freedom). The decomposed representation has w (1 DoF) + b1 (2
  DoF) + b2 (2 DoF) = 5 DoF. Same dimensionality, so you're right — the decomposition
   is just a reparameterization, not a compression. Both contain identical
  information.

  But the key prediction is about geometry, not dimensionality. The mapping between
  the two parameterizations is nonlinear:
  - Joint → factored: w = sum(b[0:3]) is linear, but b1_normalized = b[0:3] / w
  requires division
  - Factored → joint: b_joint[0:3] = w * b1 requires multiplication

  This means a linear probe can distinguish the two. If the model linearly encodes
  the joint beliefs, you can linearly recover w (it's just a sum) and the
  unnormalized component beliefs (just slice), but you cannot linearly recover the
  normalized per-component beliefs. Conversely, if the model linearly encodes (w, b1,
   b2) in orthogonal subspaces, you cannot linearly recover the joint beliefs (that
  requires multiplication).

  What the current experiment actually tests: The current probe fits a linear map
  from activations to the 6D joint belief vector, then decomposes post-hoc. This
  doesn't distinguish joint vs factored encoding — if the joint probe is good, the
  post-hoc decomposition will look reasonable regardless, because you're applying the
   nonlinear decomposition to the probe output, not testing whether the components
  are individually linearly accessible.

  What would make the prediction sharper: Separate linear probes directly for each
  component:
  1. Probe activations → w (1D)
  2. Probe activations → b1_normalized (3D, constrained to simplex)
  3. Probe activations → b2_normalized (3D, constrained to simplex)

  Then compare: is the total MSE from probes (1)+(2)+(3) better or worse than from
  the single joint probe? If factored is better, the model is using a decomposed
  representation. If joint is better, it's using the direct parameterization.

  How the union case differs from Shai 2026's product case: In the product case (b1 ⊗
   b2), recovering factors from the joint requires factoring a tensor product —
  deeply nonlinear, so factored vs joint is a strong prediction. In the union case
  (w·b1 ⊕ (1-w)·b2), the joint is a direct sum, so the unnormalized components are
  already linearly accessible. The prediction is weaker but still meaningful for the
  normalized beliefs.

  The more interesting predictions for the union case are probably:
  1. Geometric: Do the probed beliefs form two distinct simplex triangles that
  separate over sequence position, mirroring the bifurcation of w?
  2. Positional: Does effective dimensionality decrease later in the sequence as w →
  0 or 1 and only one component's beliefs remain relevant?
  3. Mechanistic: Are there attention heads that separately update each component's
  beliefs, with a separate mechanism for updating w?

  Prediction (2) is particularly distinctive to non-ergodicity — in the product case,
   all factors are always active, but here the model should "collapse" to a
  lower-dimensional representation once it resolves which component is active.

  Want me to add the direct factored probes so we can test the representational
  prediction properly?
