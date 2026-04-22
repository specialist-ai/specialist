"""Per-tracker offline visual-inertial bundle adjustment.

Consumes one Insta360 X3 ``.insv`` file and emits an optimized keyframe
trajectory plus sparse 3D landmarks and IMU biases. The output composes
into the existing cross-tracker factor graph as a subgraph. Offline /
batch; not real-time.
"""
