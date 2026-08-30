"""The dataset contract: what a recording holds and how to address it.

``schema`` names the entities and the timelines they are indexed by, ``poses``
and ``fiducials`` hold the conventions stored values follow, and ``read`` opens
a layer as numpy. Only ``read`` imports the rerun SDK.
"""

from specialist_dataset import fiducials as fiducials
from specialist_dataset import poses as poses
from specialist_dataset import schema as schema
