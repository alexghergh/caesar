import copy
import copy
import math

from dataclasses import dataclass, field
from typing import Optional

from KernelBenchInternal.eval import KernelExecResult


@dataclass
class MCTSNode:
    node_id: int
    parent_id: Optional[int]
    children: list[int] = field(default_factory=list)

    # kernel + feedback
    kernel_code: str = ""
    eval_result: KernelExecResult | None = None
    compile_summary: dict | None = None
    runtime_summary: dict | None = None
    profiler_summary: dict | None = None

    # stats
    reward: float = 0.0
    visits: int = 0
    total_reward: float = 0.0

    # branch-local memory
    prompt_messages: list = field(default_factory=list)
    coding_messages: list = field(default_factory=list)

    def mean_reward(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits


class MCTSTree:
    def __init__(self, max_children: int = 3, exploration: float = 1.4):
        self.max_children = max_children
        self.exploration = exploration
        self.nodes: dict[int, MCTSNode] = {}
        self.root_id = 0
        self.next_id = 1
        self.nodes[self.root_id] = MCTSNode(node_id=self.root_id, parent_id=None)

    def get_node(self, node_id: int) -> MCTSNode:
        return self.nodes[node_id]

    def is_fully_expanded(self, node: MCTSNode) -> bool:
        return len(node.children) >= self.max_children

    def select(self) -> tuple[list[int], int]:
        """
        Select a node to expand using UCT. Returns (path, node_id).
        """
        path = [self.root_id]
        node = self.nodes[self.root_id]

        while node.children and self.is_fully_expanded(node):
            node = self._uct_select(node)
            path.append(node.node_id)

        return path, node.node_id

    def _uct_select(self, node: MCTSNode) -> MCTSNode:
        best_score = -float("inf")
        best_child = None

        for child_id in node.children:
            child = self.nodes[child_id]
            if child.visits == 0:
                score = float("inf")
            else:
                score = (
                    child.mean_reward()
                    + self.exploration
                    * math.sqrt(math.log(node.visits + 1) / child.visits)
                )
            if score > best_score:
                best_score = score
                best_child = child

        return best_child if best_child is not None else node

    def add_child(self, parent_id: int) -> int:
        parent = self.nodes[parent_id]
        child_id = self.next_id
        self.next_id += 1

        child = MCTSNode(node_id=child_id, parent_id=parent_id)
        child.prompt_messages = copy.deepcopy(parent.prompt_messages)
        child.coding_messages = copy.deepcopy(parent.coding_messages)

        parent.children.append(child_id)
        self.nodes[child_id] = child
        return child_id

    def backprop(self, path: list[int], reward: float) -> None:
        for node_id in path:
            node = self.nodes[node_id]
            node.visits += 1
            node.total_reward += reward

    def best_node(self) -> MCTSNode | None:
        best = None
        best_score = -float("inf")
        for node in self.nodes.values():
            if node.eval_result is None:
                continue
            if not node.eval_result.compiled or not node.eval_result.correctness:
                continue
            if node.reward > best_score:
                best_score = node.reward
                best = node
        return best


def extract_baseline_runtime(eval_result: KernelExecResult | None) -> float | None:
    if eval_result is None:
        return None

    stats = getattr(eval_result, "runtime_stats", None) or {}
    for key in (
        "baseline_runtime_ms",
        "baseline_ms",
        "reference_runtime_ms",
        "ref_runtime_ms",
        "original_runtime_ms",
        "original_runtime",
    ):
        value = stats.get(key)
        if value is not None and value > 0:
            return float(value)
    return None


def compute_reward(
    eval_result: KernelExecResult | None,
    baseline_ms: float | None = None,
) -> float:
    if eval_result is None or not eval_result.compiled:
        return 0.0

    reward = 1.0
    if not eval_result.correctness:
        return reward

    reward += 2.0

    runtime = getattr(eval_result, "runtime", None)
    if runtime is None or runtime <= 0:
        return reward

    baseline = baseline_ms or extract_baseline_runtime(eval_result)
    if baseline is None or baseline <= 0:
        return reward

    speedup = baseline / runtime
    return reward + speedup
