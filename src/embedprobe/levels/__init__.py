"""The four diagnostic levels.

Each level is a pure function over a precomputed similarity matrix (and texts /
topic labels where needed), returning a dict with two keys:

- ``"metrics"``: flat, JSON-serializable summary numbers
- ``"data"``: richer arrays/records used for plotting and drill-down
"""

from embedprobe.levels.level0 import signal_to_noise
from embedprobe.levels.level1 import retrieval_metrics
from embedprobe.levels.level2 import topic_structure
from embedprobe.levels.level3 import error_taxonomy

__all__ = ["signal_to_noise", "retrieval_metrics", "topic_structure", "error_taxonomy"]
