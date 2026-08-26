"""The corpus contract: what a recording holds and how to address it.

``schema`` names the entities and the timelines they are indexed by, ``poses``
and ``fiducials`` hold the conventions stored values follow, and ``read`` opens
a layer as numpy. Only ``read`` imports the rerun SDK.
"""

from specialist_corpus import fiducials as fiducials
from specialist_corpus import poses as poses
from specialist_corpus import schema as schema
