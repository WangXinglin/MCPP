from typing import Dict, List, Optional, Tuple
import random
import pandas as pd

from .types import DEFAULT_MODEL_ID, TaskTemplate


class TaskLibrary:
    """
    Load empirically estimated rollout statistics from CSV.

    Required columns:
      success_prob, lat_mean, lat_std, cost_mean

    Optional columns (ignored if present):
      task_id, task_type
    """

    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        required = {"success_prob", "lat_mean", "lat_std", "cost_mean"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.templates: List[TaskTemplate] = []
        self.templates_by_model: Dict[str, List[TaskTemplate]] = {}
        self.is_model_aware = "model_id" in df.columns

        for _, row in df.iterrows():
            tpl = TaskTemplate(
                success_prob=float(row["success_prob"]),
                lat_mean=float(row["lat_mean"]),
                lat_std=max(float(row["lat_std"]), 1e-6),
                cost_mean=max(float(row["cost_mean"]), 1e-9),
                token_cost_mean=(
                    None
                    if "token_cost_mean" not in df.columns or pd.isna(row["token_cost_mean"])
                    else max(float(row["token_cost_mean"]), 1e-9)
                ),
            )
            self.templates.append(tpl)
            model_id = str(row["model_id"]) if self.is_model_aware else DEFAULT_MODEL_ID
            self.templates_by_model.setdefault(model_id, []).append(tpl)

        if not self.templates:
            raise ValueError("Task library is empty.")
        for model_id, templates in self.templates_by_model.items():
            if not templates:
                raise ValueError(f"Task library model_id={model_id!r} has no templates.")

        self.model_ids: Tuple[str, ...] = tuple(self.templates_by_model.keys())

    def sample_template(self, rng: Optional[random.Random] = None) -> TaskTemplate:
        """English note."""
        rng = rng or random
        if not self.is_model_aware:
            return rng.choice(self.templates)
        preferred_model = DEFAULT_MODEL_ID if DEFAULT_MODEL_ID in self.templates_by_model else self.model_ids[0]
        return rng.choice(self.templates_by_model[preferred_model])

    def sample_model_templates(self, rng: Optional[random.Random] = None) -> Dict[str, TaskTemplate]:
        """Sample one template per model_id from a model-aware library."""
        rng = rng or random
        out: Dict[str, TaskTemplate] = {}
        for model_id, templates in self.templates_by_model.items():
            if not templates:
                raise ValueError(f"Task library model_id={model_id!r} has no templates.")
            out[str(model_id)] = rng.choice(templates)
        return out
