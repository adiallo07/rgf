import re
import os
import json
import pandas as pd
from tasks.base import Task, DATA_PATH
from uop import VerifiedThoughtNode
from chat_utils import import_prompts_by_task


class PenguinInATableTask(Task):
    """
    Input (x)   : A descriptive table of penguins and a question about the data.
    Output (y)  : The correct answer to the question based on the table.
    Reward (r)  : 0 or 1, depending on whether the answer is correct.
    
    Input Example:
        Here is a table of penguins:
        name, age, height (cm), weight (kg)
        Louis, 7, 50, 11
        Bernard, 5, 80, 13
        Vincent, 9, 60, 11
        Gwen, 8, 70, 15
        Question: What is the weight of Bernard?
    
    Output Example:
        Bernard weighs 13 kg.
    """

    def __init__(self, args, file='task.json'):
        """
        Initializes the PenguinTask.

        Args:
            args: Arguments containing task-specific configurations.
            file (str): The name of the CSV file containing penguin tasks.
        """
        # Initialize the parent Task class
        super().__init__()
        # Update the object's dictionary with arguments (if any)
        self.__dict__.update(vars(args))
        path = os.path.join(DATA_PATH, 'penguin', file)
        # Load the 'Tasks' column from the CSV as a list of dictionaries
        data = json.load(open(path))
        self.data = data['examples']
        self.value_cache = {}
        self.steps = 2  # Typically, answering questions may require fewer steps
        self.stops = ['\n'] * 1  # Each answer ends with a newline
        # Import penguin-specific prompts
        self.prompts = import_prompts_by_task("penguin")
        # Configuration flags
        self.free_answer = False
        self.max_turn = 5  # Maximum number of iterations for answer refinement
        self.task_prefix = data['task_prefix']
        self.items = 4 #there are 4 penguins
        self.set = []
        self.threshold = 0.7

    def __len__(self) -> int:
        print ('length: ', len(self.data))
        """
        Returns the total number of penguin tasks.

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
        return self.task_prefix  + input_prompt, target_answer

    def test_output(self, idx: int, output: str):
        """
        Evaluates the generated answer for correctness.

        Args:
            idx (int): The index of the current task.
            output (str): The generated answer from the language model.

        Returns:
            dict: {'r': 1} if the answer is correct, else {'r': 0}.
        """
        target = self.data[idx]['target'].lower().strip()
        model_output = output.lower().strip()
        
        # Normalize whitespace and punctuation for comparison
        target_normalized = re.sub(r'\s+', ' ', target)
        model_output_normalized = re.sub(r'\s+', ' ', model_output)
        
        # Check for exact match
        if target_normalized == model_output_normalized:
            return {'r': 1}
        else:
            # Optional: Implement more sophisticated comparison (e.g., semantic similarity)
            print(f"Expected: {target_normalized}")
            print(f"Got: {model_output_normalized}")
            return {'r': 0}

    def create_root(self, root=None):
        """
        Initializes the root node for the conversation tree.

        Args:
            root (VerifiedThoughtNode, optional): An existing root node.
                                                 If None, a new root node is created.
        """
        if not root:
            self.root = VerifiedThoughtNode("ROOT", True, self.set, None, self.performer_model)
        else:
            root.set_config(self.n_extend_layers, not self.none_acc_reward, self.expected_reward_method)
            self.root = root


# Example usage:
# Assuming you have a 'penguin.csv' with a 'Tasks' column where each task is a dictionary
# containing 'input' (the table and question) and 'target' (the correct answer).

if __name__ == "__main__":
    # Sample arguments object; replace with actual arguments as needed
    class Args:
        performer_model = "gpt-4"
        teacher_model = "gpt-4"
        n_extend_layers = 2
        none_acc_reward = False
        expected_reward_method = "avg"

    args = Args()
    penguin_task = PenguinTask(args, file='penguin.csv')
    print(f"Total Penguin Tasks: {len(penguin_task)}")

    # Example Task
    idx = 0  # Index of the task to process
    input_prompt, target_answer = penguin_task.get_input(idx)
    print("Sample Input Prompt:")
    print(input_prompt)

    # Example Output (replace with actual model-generated answer)
    sample_output = "Bernard weighs 13 kg."
    
    evaluation = penguin_task.test_output(idx, sample_output)
    print(f"Evaluation: {evaluation}")