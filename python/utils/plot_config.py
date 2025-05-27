from __future__ import annotations
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

@dataclass
class PlotConfig:
    """Plot Configuration."""
    title: str
    ylabel: str
    xlabel: str = "Time [s]"
    num_subplots: int = 1
    subplot_kwargs: Dict[str, bool] | None = None
    keys: List[str] | None = None
    colors: List[str] | None = None
    linestyles: List[str] | None = None
    linewidths: List[int] | None = None
