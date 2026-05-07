from typing import Dict, List, Optional
from collections import deque
import json
import random

from .types import DEFAULT_MODEL_ID, NodeSpec, TaskTemplate
from .task_library import TaskLibrary


def _template_to_dict(tpl: TaskTemplate) -> dict:
    out = {
        "success_prob": tpl.success_prob,
        "lat_mean": tpl.lat_mean,
        "lat_std": tpl.lat_std,
        "cost_mean": tpl.cost_mean,
    }
    if tpl.token_cost_mean is not None:
        out["token_cost_mean"] = tpl.token_cost_mean
    return out


def _template_from_dict(data: dict) -> TaskTemplate:
    return TaskTemplate(
        success_prob=data["success_prob"],
        lat_mean=data["lat_mean"],
        lat_std=data["lat_std"],
        cost_mean=data["cost_mean"],
        token_cost_mean=data.get("token_cost_mean"),
    )


def _assigned_template_for_models(model_templates: Dict[str, TaskTemplate]) -> TaskTemplate:
    if DEFAULT_MODEL_ID in model_templates:
        return model_templates[DEFAULT_MODEL_ID]
    first_model_id = next(iter(model_templates))
    return model_templates[first_model_id]


class DAG:
    def __init__(self, nodes: Dict[int, NodeSpec]):
        self.nodes = nodes
        self.node_ids = sorted(nodes.keys())

    def roots(self) -> List[int]:
        return [nid for nid, n in self.nodes.items() if len(n.parents) == 0]

    def sinks(self) -> List[int]:
        return [nid for nid, n in self.nodes.items() if len(n.children) == 0]

    def topological_order(self) -> List[int]:
        indeg = {nid: len(n.parents) for nid, n in self.nodes.items()}
        q = deque([nid for nid, d in indeg.items() if d == 0])
        order = []
        while q:
            u = q.popleft()
            order.append(u)
            for v in self.nodes[u].children:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        if len(order) != len(self.nodes):
            raise ValueError("Graph is not a DAG.")
        return order

    # ---- English note / English note / English note ----

    def to_topo_dict(self, dag_kind: str = None) -> dict:
        """
        English note DAG English note(English note).

        English note DAG English note,English note DAG English note.
        English note dag_kind English note,English note.
        """
        nodes_list = []
        for nid in sorted(self.nodes.keys()):
            node = self.nodes[nid]
            nodes_list.append({
                "node_id": node.node_id,
                "parents": list(node.parents),
                "children": list(node.children),
                "layer": node.layer,
            })
        d = {"n_nodes": len(self.nodes), "nodes": nodes_list}
        if dag_kind is not None:
            d["dag_kind"] = dag_kind
        return d

    @classmethod
    def from_topo_dict(cls, data: dict) -> "DAG":
        """English note DAG English note(assigned_template English note None)."""
        nodes: Dict[int, NodeSpec] = {}
        for nd in data["nodes"]:
            nodes[nd["node_id"]] = NodeSpec(
                node_id=nd["node_id"],
                parents=list(nd["parents"]),
                children=list(nd["children"]),
                layer=nd["layer"],
                assigned_template=None,  # type: ignore
            )
        return cls(nodes)

    def with_templates(self, lib: TaskLibrary, rng: random.Random) -> "DAG":
        """
        English note,English note DAG English note.

        English note DAG English note,English note
        (English note:subtask English note).
        """
        new_nodes: Dict[int, NodeSpec] = {}
        for nid in sorted(self.nodes.keys()):
            node = self.nodes[nid]
            if getattr(lib, "is_model_aware", False):
                model_templates = lib.sample_model_templates(rng)
                assigned_template = _assigned_template_for_models(model_templates)
                new_nodes[nid] = NodeSpec(
                    node_id=node.node_id,
                    parents=list(node.parents),
                    children=list(node.children),
                    layer=node.layer,
                    assigned_template=assigned_template,
                    model_templates=model_templates,
                )
                continue
            new_nodes[nid] = NodeSpec(
                node_id=node.node_id,
                parents=list(node.parents),
                children=list(node.children),
                layer=node.layer,
                assigned_template=lib.sample_template(rng),
            )
        return DAG(new_nodes)

    def to_full_dict(self, dag_kind: str = None) -> dict:
        """
        English note DAG English note(English note + English note).

        English note lat_mean, lat_std, success_prob, cost_mean,
        English note DAG English note,English note TaskLibrary English note.
        """
        nodes_list = []
        for nid in sorted(self.nodes.keys()):
            node = self.nodes[nid]
            tpl = node.assigned_template
            nd = {
                "node_id": node.node_id,
                "parents": list(node.parents),
                "children": list(node.children),
                "layer": node.layer,
            }
            if tpl is not None:
                nd["template"] = _template_to_dict(tpl)
            if node.empirical_samples is not None:
                nd["empirical_samples"] = node.empirical_samples
            if node.model_templates:
                nd["model_templates"] = {
                    str(model_id): _template_to_dict(model_tpl)
                    for model_id, model_tpl in node.model_templates.items()
                }
            if node.empirical_samples_by_model:
                nd["empirical_samples_by_model"] = {
                    str(model_id): samples
                    for model_id, samples in node.empirical_samples_by_model.items()
                }
            if node.legacy_template_model is not None:
                nd["legacy_template_model"] = node.legacy_template_model
            nodes_list.append(nd)
        d = {"n_nodes": len(self.nodes), "nodes": nodes_list}
        if dag_kind is not None:
            d["dag_kind"] = dag_kind
        return d

    @classmethod
    def from_full_dict(cls, data: dict) -> "DAG":
        """English note DAG(English note + English note + English note empirical English note)."""
        nodes: Dict[int, NodeSpec] = {}
        for nd in data["nodes"]:
            tpl_data = nd.get("template")
            tpl = _template_from_dict(tpl_data) if tpl_data else None
            model_templates_data = nd.get("model_templates")
            model_templates = None
            if model_templates_data:
                model_templates = {
                    str(model_id): _template_from_dict(model_tpl)
                    for model_id, model_tpl in model_templates_data.items()
                }
                if tpl is None:
                    tpl = _assigned_template_for_models(model_templates)
            empirical_samples_by_model = nd.get("empirical_samples_by_model")
            if empirical_samples_by_model:
                empirical_samples_by_model = {
                    str(model_id): samples
                    for model_id, samples in empirical_samples_by_model.items()
                }
            nodes[nd["node_id"]] = NodeSpec(
                node_id=nd["node_id"],
                parents=list(nd["parents"]),
                children=list(nd["children"]),
                layer=nd["layer"],
                assigned_template=tpl,
                empirical_samples=nd.get("empirical_samples"),
                model_templates=model_templates,
                empirical_samples_by_model=empirical_samples_by_model,
                legacy_template_model=nd.get("legacy_template_model"),
            )
        return cls(nodes)

    @staticmethod
    def save_pool_jsonl(pool: List[dict], filepath: str) -> None:
        """English note DAG English note JSONL English note(English note DAG)."""
        with open(filepath, "w") as f:
            for entry in pool:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    @staticmethod
    def load_pool_jsonl(filepath: str) -> List[dict]:
        """English note JSONL English note DAG English note."""
        pool = []
        with open(filepath, "r") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if line:
                    try:
                        pool.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        preview = line[:240]
                        if len(line) > 240:
                            preview += "..."
                        raise ValueError(
                            f"Invalid JSONL in DAG pool {filepath!r} at line {line_no}, "
                            f"column {exc.colno}: {exc.msg}. Line preview: {preview!r}"
                        ) from exc
        return pool

    @staticmethod
    def save_topos(topos: List[dict], filepath: str) -> None:
        """English note DAG English note JSON English note."""
        with open(filepath, "w") as f:
            json.dump(topos, f)

    @staticmethod
    def load_topos(filepath: str) -> List[dict]:
        """English note JSON English note DAG English note."""
        with open(filepath, "r") as f:
            return json.load(f)

    # ---- English note ----

    def induced_remaining_subgraph(self, completed_nodes: set[int]) -> "DAG":
        mapping = {}
        for nid, node in self.nodes.items():
            if nid in completed_nodes:
                continue
            parents = [p for p in node.parents if p not in completed_nodes]
            children = [c for c in node.children if c not in completed_nodes]
            mapping[nid] = NodeSpec(
                node_id=nid,
                parents=parents,
                children=children,
                layer=node.layer,
                assigned_template=node.assigned_template,
                empirical_samples=node.empirical_samples,
                model_templates=node.model_templates,
                empirical_samples_by_model=node.empirical_samples_by_model,
                legacy_template_model=node.legacy_template_model,
            )
        return DAG(mapping)


class DAGGenerator:
    def __init__(self, task_library: TaskLibrary, seed: int = 0):
        self.lib = task_library
        self.rng = random.Random(seed)

    def generate_layered_random_dag(
        self,
        n_nodes: int,
        n_layers: int = 4,
        edge_prob: float = 0.25,
    ) -> DAG:
        if n_layers < 2:
            raise ValueError("n_layers must be >= 2")
        if n_nodes < n_layers:
            raise ValueError("n_nodes must be >= n_layers")

        layer_sizes = [1] * n_layers
        remaining = n_nodes - n_layers
        for _ in range(remaining):
            layer_sizes[self.rng.randrange(n_layers)] += 1

        layers: List[List[int]] = []
        cur = 0
        for sz in layer_sizes:
            layers.append(list(range(cur, cur + sz)))
            cur += sz

        nodes: Dict[int, NodeSpec] = {}
        for layer_idx, layer_nodes in enumerate(layers):
            for nid in layer_nodes:
                nodes[nid] = NodeSpec(
                    node_id=nid,
                    parents=[],
                    children=[],
                    layer=layer_idx,
                    assigned_template=None,  # type: ignore
                )

        for i in range(n_layers - 1):
            for u in layers[i]:
                connected = False
                for j in range(i + 1, n_layers):
                    for v in layers[j]:
                        if self.rng.random() < edge_prob:
                            nodes[u].children.append(v)
                            nodes[v].parents.append(u)
                            connected = True
                if not connected:
                    v = self.rng.choice(layers[i + 1])
                    if v not in nodes[u].children:
                        nodes[u].children.append(v)
                        nodes[v].parents.append(u)

        for layer_idx in range(1, n_layers):
            previous_nodes = [x for layer in layers[:layer_idx] for x in layer]
            for v in layers[layer_idx]:
                if len(nodes[v].parents) == 0:
                    u = self.rng.choice(previous_nodes)
                    nodes[u].children.append(v)
                    nodes[v].parents.append(u)

        # English note
        for nid, node in nodes.items():
            if getattr(self.lib, "is_model_aware", False):
                model_templates = self.lib.sample_model_templates(self.rng)
                node.model_templates = model_templates
                node.assigned_template = _assigned_template_for_models(model_templates)
            else:
                node.assigned_template = self.lib.sample_template(self.rng)

        return DAG(nodes)

    def generate_chain(self, n_nodes: int) -> DAG:
        nodes: Dict[int, NodeSpec] = {}
        for i in range(n_nodes):
            parents = [i - 1] if i > 0 else []
            children = [i + 1] if i < n_nodes - 1 else []
            if getattr(self.lib, "is_model_aware", False):
                model_templates = self.lib.sample_model_templates(self.rng)
                tpl = _assigned_template_for_models(model_templates)
                nodes[i] = NodeSpec(i, parents, children, i, tpl, model_templates=model_templates)
            else:
                tpl = self.lib.sample_template(self.rng)
                nodes[i] = NodeSpec(i, parents, children, i, tpl)
        return DAG(nodes)

    def generate_fork_join(
        self,
        width: int,
        middle_depth: int = 1,
    ) -> DAG:
        nodes: Dict[int, NodeSpec] = {}
        nid = 0
        root = nid
        if getattr(self.lib, "is_model_aware", False):
            model_templates = self.lib.sample_model_templates(self.rng)
            nodes[root] = NodeSpec(
                root,
                [],
                [],
                0,
                _assigned_template_for_models(model_templates),
                model_templates=model_templates,
            )
        else:
            nodes[root] = NodeSpec(root, [], [], 0, self.lib.sample_template(self.rng))
        nid += 1

        branch_nodes = []
        for _ in range(width):
            prev = root
            for d in range(middle_depth):
                cur = nid
                if getattr(self.lib, "is_model_aware", False):
                    model_templates = self.lib.sample_model_templates(self.rng)
                    tpl = _assigned_template_for_models(model_templates)
                    nodes[cur] = NodeSpec(
                        cur,
                        [prev],
                        [],
                        d + 1,
                        tpl,
                        model_templates=model_templates,
                    )
                else:
                    tpl = self.lib.sample_template(self.rng)
                    nodes[cur] = NodeSpec(cur, [prev], [], d + 1, tpl)
                nodes[prev].children.append(cur)
                prev = cur
                nid += 1
            branch_nodes.append(prev)

        join = nid
        if getattr(self.lib, "is_model_aware", False):
            model_templates = self.lib.sample_model_templates(self.rng)
            nodes[join] = NodeSpec(
                join,
                list(branch_nodes),
                [],
                middle_depth + 1,
                _assigned_template_for_models(model_templates),
                model_templates=model_templates,
            )
        else:
            nodes[join] = NodeSpec(
                join,
                list(branch_nodes),
                [],
                middle_depth + 1,
                self.lib.sample_template(self.rng),
            )
        for b in branch_nodes:
            nodes[b].children.append(join)

        return DAG(nodes)
