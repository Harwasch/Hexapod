# Minnetonka tree skeleton (test fixture)

`rig.json` is the 200-joint skeleton `tools/captures/skeleton.py` extracted from a Gaussian
splat reconstruction of the Minnetonka tree (6.0 m, `sites/minnetonka-tree/`), reduced for
testing: positions rounded to 0.1 mm, the motion pointer dropped, no splats.

The source imagery is **Matthew Guertin's tree photogrammetry dataset**
(<https://github.com/Matt1Up/tree-photogrammetry-dataset>), licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The skeleton is a derived work of
that dataset and is shared under the same licence, with attribution to Matthew Guertin.

It is used only as a test case — a real, fragmented, zigzagging extracted skeleton — for rules
that are derived from any rig's own geometry. Nothing in the motion model is fitted to it.
