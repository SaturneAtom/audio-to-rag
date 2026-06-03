import json
import numpy as np
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional


@dataclass
class Chunk:
    """Chunk de conversation avec métadonnées sémantiques"""
    id: int
    start_time: float
    end_time: float
    text: str
    embedding: Optional[np.ndarray] = None
    speakers_info: List[Dict[str, Any]] = field(default_factory=list)
    is_topic_transition: bool = False
    transition_score: float = 1.0
    cluster_id: int = -1
    theme: str = ""
    reasoning: str = ""
    nova_chunking: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if "embedding" in result:
            del result["embedding"]
        return result


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)
