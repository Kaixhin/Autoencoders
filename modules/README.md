modules
=======

The modules produce multivariate samples from probability distributions, using the same parameters during initialisation as used in the standard torch random functions. The size of the input tensor dictates the size of the output (sample) tensor, allowing this to change during runtime. Correspondingly, a zero tensor is returned in the backwards pass.
