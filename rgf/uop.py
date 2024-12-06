import numpy as np
from copy import deepcopy

from chat_utils import (
    ques_and_cls_given_items,
    cls_given_repo,
    initialize_open_set,
    renew_open_set,
)
from models import get_response_method


class VerifiedThoughtNode:
    def __init__(
        self,
        solution,
        is_valid,
        items,
        parent: 'VerifiedThoughtNode' = None,
        model="gpt-4",
        feedback=None,
    ):
        """
        Initializes a node representing a Performer's solution and Teacher's feedback.

        Args:
            solution (str): The Performer's proposed solution.
            is_valid (bool): Whether the solution adheres to the rules.
            items (list): Relevant items or information related to the solution.
            parent (VerifiedThoughtNode, optional): Parent node in the solution tree.
            model (str, optional): The LLM model used.
            feedback (str, optional): Feedback from the Teacher.
        """
        self.children = []
        self.solution = solution
        self.is_valid = is_valid  # True for "YES" and False for "No"
        self.feedback = feedback
        self.items = items
        self.parent = parent
        print ('----', self.parent)
        self.depth = self.parent.depth + 1 if self.parent else 0
        self.model = model
        self.n_extend_layers = -1
        self.accumulation = True
        self.expected_method = 'avg'
        self.print_node()

    def set_config(self, n_extend_layers: int, none_acc: bool, exp: str):
        """
        Sets configuration for reward calculations.

        Args:
            n_extend_layers (int): Number of layers to extend.
            none_acc (bool): Whether to accumulate rewards.
            exp (str): Expected method ('avg' or 'max').
        """
        self.n_extend_layers = n_extend_layers
        self.accumulation = not none_acc
        self.expected_method = exp

    def _create_children_nodes(self, task, items: list, n, asked_solutions: list = None):
        """
        Generates child nodes based on Performer's solutions and Teacher's feedback.

        Args:
            task: The task configuration.
            items (list): Current items or information.
            n (int): Number of children to create.
            asked_solutions (list, optional): Solutions already proposed.
        """
        items = list(set(items))
        if self.is_terminal:
            return
        # Generate possible solution branches based on current items
        ans = ques_and_cls_given_items(task, items, n, asked_solutions)
        if not ans:
            return
        for a in ans:
            # Child node where solution is valid
            child_valid = VerifiedThoughtNode(
                solution=a["solution_valid"],
                is_valid=True,
                items=a["items_valid"],
                parent=self,
                model=self.model,
                feedback="Valid Solution."
            )
            # Child node where solution is invalid
            child_invalid = VerifiedThoughtNode(
                solution=a["solution_invalid"],
                is_valid=False,
                items=a["items_invalid"],
                parent=self,
                model=self.model,
                feedback="Invalid Solution: Does not adhere to the rules."
            )
            self.children.extend([child_valid, child_invalid])

    def find_children(self, task=None, n=None):
        """
        Retrieves or generates child nodes.

        Args:
            task: The task configuration.
            n (int, optional): Number of children to find.

        Returns:
            list or None: List of child nodes or None if terminal.
        """
        if self.is_terminal:
            return None
        if task and n and (not self.children or len(self.children) < n):
            asked_solutions = [child.solution for child in self.children] if self.children else []
            self._create_children_nodes(task, self.items, n - len(asked_solutions), asked_solutions)
        return self.children

    def find_children_sep(self, task=None, n=None, prune=0):
        """
        Separates child nodes and optionally prunes them.

        Args:
            task: The task configuration.
            n (int, optional): Number of children to find.
            prune (int, optional): Pruning parameter.

        Returns:
            list or None: List of separated child nodes or None.
        """
        _children = self.find_children(task, n)
        return_list = [c for c in _children] if _children else None
        if prune < 0 and return_list:
            # Prune the list based on reward
            return_list = sorted(return_list, key=lambda x: x.composite_reward, reverse=True)[:int(-prune * len(return_list))]
        return return_list

    def handle_self_repo(self, task, repo, translate=False):
        """
        Handles self-reporting from the Performer.

        Args:
            task: The task configuration.
            repo (str): Repository information.
            translate (bool, optional): Whether to translate the repository.

        Returns:
            VerifiedThoughtNode: Updated node after handling self-report.
        """
        if task.open_set_size > 0:
            a = initialize_open_set(task, repo)
            node_valid = VerifiedThoughtNode("self-report", True, a, parent=self, model=self.model)
            node_invalid = VerifiedThoughtNode("self-report", False, [], parent=self, model=self.model)
        else:
            a = cls_given_repo(task, self.items, repo, translate, self_repo=True)
            node_valid = VerifiedThoughtNode("self-report", True, a["items_yes"], parent=self, model=self.model)
            node_invalid = VerifiedThoughtNode("self-report", False, a["items_no"], parent=self, model=self.model)

        exist_leaves = []
        for c in self.children:
            exist_leaves.extend(c)
        if node_valid in exist_leaves:
            return exist_leaves[exist_leaves.index(node_valid)]
        self.children.append((node_valid, node_invalid))
        return node_valid

    def handle_feedback(self, task, feedback, translate=False):
        """
        Handles Teacher's feedback to refine the solution.

        Args:
            task: The task configuration.
            feedback (str): Feedback from the Teacher.
            translate (bool, optional): Whether to translate the feedback.

        Returns:
            VerifiedThoughtNode: Updated node based on feedback.
        """
        if task.open_set_size > 0:
            a = renew_open_set(task, feedback, self.items)
            node_valid = VerifiedThoughtNode("renew", True, a, parent=self, model=self.model)
            node_invalid = VerifiedThoughtNode("renew", False, [], parent=self, model=self.model)
        else:
            a = cls_given_repo(task, self.items, feedback, translate, self_repo=False)
            node_valid = VerifiedThoughtNode("renew", True, a["items_yes"], parent=self, model=self.model)
            node_invalid = VerifiedThoughtNode("renew", False, a["items_no"], parent=self, model=self.model)

        exist_leaves = []
        for c in self.children:
            exist_leaves.extend(c)
        if node_valid in exist_leaves:
            return exist_leaves[exist_leaves.index(node_valid)]
        self.children.append((node_valid, node_invalid))
        return node_valid

    def ans2node(self, answer: bool):
        """
        Retrieves the corresponding child node based on the Teacher's answer.

        Args:
            answer (bool): Teacher's validation (True for valid, False for invalid).

        Returns:
            VerifiedThoughtNode or None: The corresponding child node.
        """
        return self if self.is_valid == answer else next(
            (
                child
                for child in self.children
                if child.is_valid == answer
            ),
            None
        )

    @staticmethod
    def rule_adherence_reward(num_rules_met, total_rules, lamb=0.5):
        """
        Calculates a reward based on rule adherence.

        Args:
            num_rules_met (int): Number of rules met by the solution.
            total_rules (int): Total number of rules.
            lamb (float, optional): Lambda parameter to scale the reward.

        Returns:
            float: Calculated reward.
        """
        adherence_ratio = num_rules_met / total_rules
        # Using a scaled sigmoid function for smooth reward
        reward = 1 / (1 + np.exp(-lamb * (adherence_ratio - 0.5)))
        return reward

    @staticmethod
    def solution_correctness_reward(is_correct, lamb=1.0):
        """
        Calculates a reward based on solution correctness.

        Args:
            is_correct (bool): Whether the solution is correct.
            lamb (float, optional): Lambda parameter to scale the reward.

        Returns:
            float: Calculated reward.
        """
        return lamb if is_correct else 0

    def composite_reward(self, total_rules):
        """
        Combines multiple reward components into a composite reward.

        Args:
            total_rules (int): Total number of rules.

        Returns:
            float: Composite reward.
        """
        adherence_reward = self.rule_adherence_reward(self.count_rules_met(), total_rules)
        correctness_reward = self.solution_correctness_reward(self.is_valid)
        composite = 0.7 * adherence_reward + 0.3 * correctness_reward
        return composite

    def count_rules_met(self):
        """
        Counts the number of rules met by the solution.

        Returns:
            int: Number of rules met.
        """
        # Placeholder: Implement actual rule checking logic
        # For example, parse the solution and verify each rule
        # Here, returning a dummy value for illustration
        return len([rule for rule in getattr(self.parent, 'rules', []) if rule in self.solution])

    @staticmethod
    def accumulated_reward(node, level, accum=True):
        """
        Accumulates rewards up the node tree.

        Args:
            node (VerifiedThoughtNode): Current node.
            level (int): Current level.
            accum (bool, optional): Whether to accumulate.

        Returns:
            float: Accumulated reward.
        """
        term = 0 if (level == 1 or not accum) else node.accumulated_reward(node.parent, level - 1, accum)
        return node.composite_reward(node.parent.total_rules) + term

    @staticmethod
    def avg_expected(setting_node, child_list, n_extend_layers, level, prob, total_rules):
        """
        Calculates average expected reward.

        Args:
            setting_node (VerifiedThoughtNode): Current node.
            child_list (list): List of child nodes.
            n_extend_layers (int): Layers to extend.
            level (int): Current level.
            prob (float): Probability.
            total_rules (int): Total number of rules.

        Returns:
            float: Average expected reward.
        """
        if not child_list:
            return 0
        child_r = 0.0
        for child_node in child_list:
            child_node.set_config(setting_node.n_extend_layers, setting_node.accumulation, setting_node.expected_method)
            child_r += child_node.expected_reward(n_extend_layers, level=level + 1, total_rules=total_rules)
        return child_r * prob / len(child_list) if len(child_list) > 0 else 1

    @staticmethod
    def max_expected(setting_node, child_list, n_extend_layers, level, prob, total_rules):
        """
        Calculates maximum expected reward.

        Args:
            setting_node (VerifiedThoughtNode): Current node.
            child_list (list): List of child nodes.
            n_extend_layers (int): Layers to extend.
            level (int): Current level.
            prob (float): Probability.
            total_rules (int): Total number of rules.

        Returns:
            float: Maximum expected reward.
        """
        if not child_list:
            return 0
        child_r = 0.0
        for child_node in child_list:
            child_node.set_config(setting_node.n_extend_layers, setting_node.accumulation, setting_node.expected_method)
            child_r = max(child_node.expected_reward(n_extend_layers, level=level + 1, total_rules=total_rules), child_r)
        return child_r * prob

    def expected_reward(self, n_extend_layers, level=1, total_rules=0):
        """
        Recursively calculates the expected reward for the node.

        Args:
            n_extend_layers (int): Layers to extend.
            level (int, optional): Current level.
            total_rules (int): Total number of rules.

        Returns:
            float: Expected reward.
        """
        if not self.parent:
            return 1.0  # Root node reward

        c_1, c_2 = self.count_M_U()
        p = c_1 / (c_1 + c_2) if (c_1 + c_2) > 0 else 0.5
        partner = self.ans2node(not self.is_valid)
        if level == self.n_extend_layers - 1 or self.is_terminal or not self.children:
            return self.accumulated_reward(self, level, self.accumulation)
        else:
            expected_function = self.avg_expected if self.expected_method == 'avg' else self.max_expected
            avg_1 = expected_function(self, self.find_children_sep(), n_extend_layers, level, p, total_rules)
            avg_2 = expected_function(self, partner.find_children_sep(), n_extend_layers, level, 1 - p, total_rules)
        return (p * (self.composite_reward(total_rules) + avg_1) +
                (1 - p) * (partner.composite_reward(total_rules) + avg_2))

    @property
    def reward(self):
        """
        Computes the node's overall reward.

        Returns:
            float: Overall reward.
        """
        total_rules = getattr(self.parent, 'total_rules', 1)  # Default to 1 to avoid division by zero
        self.set_config(self.parent.n_extend_layers, self.parent.accumulation, self.parent.expected_method)
        return self.expected_reward(self.n_extend_layers, total_rules=total_rules)

    @property
    def is_terminal(self):
        """
        Determines if the node is a terminal node.

        Returns:
            bool: True if terminal, else False.
        """
        return len(self.items) <= 2

    def print_node(self):
        """
        Prints the node's information for debugging.
        """
        print(
            f"Solution: {self.solution}; Valid: {self.is_valid}; Items: {len(self.items)}; Depth: {self.depth}; Terminal: {self.is_terminal}"
        )

    def __eq__(self, other):
        """
        Checks equality based on items and depth.

        Args:
            other (VerifiedThoughtNode): Another node to compare.

        Returns:
            bool: True if equal, else False.
        """
        if isinstance(other, VerifiedThoughtNode):
            if len(self.items) != len(other.items) or self.depth != other.depth:
                return False
            for i in self.items:
                if i not in other.items:
                    return False
            return True
        else:
            return False

    def __lt__(self, other):
        """
        Less-than comparison based on reward.

        Args:
            other (VerifiedThoughtNode): Another node to compare.

        Returns:
            bool: True if self.reward < other.reward, else False.
        """
        if isinstance(other, VerifiedThoughtNode):
            return self.reward < other.reward
        raise ValueError("Comparison with non-VerifiedThoughtNode object")

    def __gt__(self, other):
        """
        Greater-than comparison based on reward.

        Args:
            other (VerifiedThoughtNode): Another node to compare.

        Returns:
            bool: True if self.reward > other.reward, else False.
        """
        if isinstance(other, VerifiedThoughtNode):
            return self.reward > other.reward
        raise ValueError("Comparison with non-VerifiedThoughtNode object")


def expand(task, root):
    """
    Expands the tree up to a specified number of layers.

    Args:
        task: The task configuration.
        root (VerifiedThoughtNode): Root node of the solution tree.

    Returns:
        list or None: List of nodes at the final layer or None.
    """
    n_layer = task.n_extend_layers
    nodes = [[] for _ in range(n_layer)]

    # Initialize first layer
    new_nodes = root.find_children_sep(task, task.n_potential_actions)
    if not new_nodes:
        return None
    nodes[0].extend(new_nodes)
    print(f"Layer 0: {len(nodes[0])} nodes")

    # Iterate through layers
    for layer in range(1, n_layer):
        nodes[layer].extend(
            new_node
            for cur_node in nodes[layer - 1]
            for new_node in (cur_node.find_children_sep(task, task.n_potential_actions, prune=task.n_pruned_nodes) or [cur_node])
        )
        if task.n_pruned_nodes > 0:
            nodes[layer] = sorted(nodes[layer], key=lambda x: x.reward, reverse=True)[: int(task.n_pruned_nodes)]
        print(f"Layer {layer}: {len(nodes[layer])} nodes")

    return nodes[n_layer - 1]


def select(task, node):
    """
    Selects the best node based on reward.

    Args:
        task: The task configuration.
        node (VerifiedThoughtNode): Current node.

    Returns:
        VerifiedThoughtNode or None: Selected node.
    """
    leaf_nodes = expand(task, node)
    candidates = node.find_children_sep()
    if not leaf_nodes or not candidates:
        return None
    # Select the node with the highest composite reward
    return max(candidates, key=lambda n: n.reward, default=None)


def renew_node_to_root(task, node, history):
    """
    Renews the node to root to explore alternative solutions.

    Args:
        task: The task configuration.
        node (VerifiedThoughtNode): Current node.
        history (list): Conversation history.

    Returns:
        VerifiedThoughtNode: Renewed node.
    """
    a = renew_open_set(task, history, node.items)
    node_valid = VerifiedThoughtNode("renew", True, a, parent=task.root, model=node.model)
    node_invalid = VerifiedThoughtNode("renew", False, [], parent=task.root, model=node.model)
    exist_leaves = []
    for c in task.root.children:
        exist_leaves.extend(c)
    if node_valid in exist_leaves:
        return exist_leaves[exist_leaves.index(node_valid)]
    task.root.children.append((node_valid, node_invalid))
    return node_valid