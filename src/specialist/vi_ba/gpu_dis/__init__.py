"""GPU (PyTorch) reimplementation of OpenCV's DIS optical flow.

Vendored from the research spike that validated the propagation-ON anti-diagonal
wavefront against the cv2 reference. Both modules keep their upstream research
style (4-space indent, untyped), so they are excluded from the formatter and the
type checker; the lint gate still reads them for undefined and unused names.
specialist.vi_ba.gpu_flow is the sole consumer and imports torch_dis_prop_fast
lazily, because torch and triton are imported at these modules' top level and
both are optional dependencies.
"""
