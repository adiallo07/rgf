import re
import os
import json
import pandas as pd
import chess
import chess.pgn
from tasks.base import Task, DATA_PATH

from chat_utils import import_prompts_by_task


class CheckmateInOneTask(Task):
    """
    Input (x)   : A series of chess moves in Standard Algebraic Notation (SAN).
    Output (y)  : The next move in SAN that results in a checkmate.
    Reward (r)  : 1 if the move results in a checkmate, else 0.

    Input Example:
        1. e4 e5
        2. Qh5 Nc6
        3. Bc4 Nf6
        4. Qxf7#

    Output Example:
        Qxf7#
    """

    def __init__(self, args, file='CheckmateInOne.jsonl'):
        """
        Initializes the CheckmateInOneTask.

        Args:
            args: Arguments containing task-specific configurations.
            file (str): The name of the JSONL file containing checkmate tasks.
        """
        # Initialize the parent Task class
        super().__init__()
        # Update the object's dictionary with arguments (if any)
        self.__dict__.update(vars(args))
        # Path to the checkmate JSONL file
        path = os.path.join(DATA_PATH, 'checkmate', file)
        # Load the 'Tasks' column from the JSONL as a list of dictionaries
        self.data = self.load_tasks(path)
        self.value_cache = {}
        self.steps = 2  # Typically, answering requires fewer iterative steps
        self.stops = ['\n']  # Each answer ends with a newline
        # Import checkmate-specific prompts
        self.prompts = import_prompts_by_task("checkmate")
        # Configuration flags
        self.free_answer = False
        self.max_turn = 5  # Maximum number of iterations for answer refinement
        self.threshold = 0.7
        self.task_inform = True

    def load_tasks(self, path):
        """
        Loads tasks from a JSONL file.

        Args:
            path (str): Path to the JSONL file.

        Returns:
            list: List of task dictionaries with 'input' and 'target'.
        """
        tasks = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                task = json.loads(line)
                tasks.append(task)
        return tasks

    def __len__(self) -> int:
        print ('length: ', len(self.data))
        """
        Returns the total number of checkmate tasks.

        Returns:
            int: Number of tasks.
        """
        return len(self.data)

    def get_input(self, idx: int) -> str:
        """
        Retrieves the input prompt for the language model based on the task index.

        Args:
            idx (int): The index of the current task.

        Returns:
            tuple: A tuple containing the input prompt and the target answer.
        """
        task = self.data[idx]
        input_prompt = task['input']
        target_answer = task['target']
        return input_prompt, target_answer

    def test_output(self, idx: int, output: str):
        """
        Evaluates the generated move for correctness.

        Args:
            idx (int): The index of the current task.
            output (str): The generated move from the language model.

        Returns:
            dict: {'r': 1} if the move results in checkmate, else {'r': 0}.
        """
        task_info = self.data[idx]
        input_moves = task_info['input']
        expected_move = task_info['target']

        # Initialize the chess board
        board = chess.Board()

        # Parse and apply moves from the input
        try:
            for move_san in self.parse_moves(input_moves):
                move = board.parse_san(move_san)
                board.push(move)
        except ValueError as e:
            print(f"Error parsing moves: {e}")
            return {'r': 0}

        # Check if the expected move is a legal move
        try:
            expected_move_obj = board.parse_san(expected_move)
        except ValueError:
            print(f"Expected move '{expected_move}' is not a legal SAN move.")
            return {'r': 0}

        # Check if the expected move leads to checkmate
        board.push(expected_move_obj)
        if board.is_checkmate():
            return {'r': 1}
        else:
            print("Generated move does not result in checkmate.")
            return {'r': 0}

    def parse_moves(self, input_moves: str):
        """
        Parses the series of moves from the input string.

        Args:
            input_moves (str): A string containing moves in SAN, possibly numbered.

        Returns:
            list: A list of move strings in SAN.
        """
        # Remove move numbers and extract move pairs
        moves = re.findall(r'\d+\.\s*([^\d]+)', input_moves)
        san_moves = []
        for move_pair in moves:
            pair = move_pair.strip().split(' ')
            san_moves.extend(pair)
        # Remove any empty strings
        san_moves = [move.strip() for move in san_moves if move.strip()]
        return san_moves

    def create_root(self, root=None):
        """
        Initializes the root node for the conversation tree.

        Args:
            root (VerifiedThoughtNode, optional): An existing root node.
                                                 If None, a new root node is created.
        """
        if not root:
            self.root = VerifiedThoughtNode("ROOT", True, [], None, self.performer_model)
        else:
            root.set_config(self.n_extend_layers, not self.none_acc_reward, self.expected_reward_method)
            self.root = root


# Example usage:
# Assuming you have a 'CheckmateInOne.jsonl' with lines like:
# {"input": "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7#", "target": "Qxf7#"}

if __name__ == "__main__":
    # Sample arguments object; replace with actual arguments as needed
    class Args:
        performer_model = "gpt-4"
        teacher_model = "gpt-4"
        n_extend_layers = 2
        none_acc_reward = False
        expected_reward_method = "avg"

    args = Args()
    checkmate_task = CheckmateInOneTask(args, file='CheckmateInOne.jsonl')
    print(f"Total CheckmateInOne Tasks: {len(checkmate_task)}")

    # Example Task
    idx = 0  # Index of the task to process
    input_prompt, target_move = checkmate_task.get_input(idx)
    print("Sample Input Prompt:")
    print(input_prompt)

    # Example Output (replace with actual model-generated move)
    sample_output = "Qxf7#"

    evaluation = checkmate_task.test_output(idx, sample_output)
    print(f"Evaluation: {evaluation}")