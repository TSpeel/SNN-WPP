Fork from MiST implementation of https://gitlab.socsci.ru.nl/stijn.groenen/snn-mst/-/tree/main?ref_type=heads
Altered to construct MaST instead.


invertMiST.py works by inverting weights in the input graph such that maximum weight edges become minumum weight edges during the pre-processing step. This is used as input to the MiST SNN implementation, and are inverted back in the output graph in 
post-processing.

createMaST.py uses an altered SNN implementation, such that edges of maximum weight are selected instead of minimum weight.
