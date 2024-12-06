import re
import os
import json
import pronouncing
from tasks.base import Task, DATA_PATH
from uop import VerifiedThoughtNode
from chat_utils import import_prompts_by_task


class SonnetWritingTask(Task):
    """
    Input (x)   : A prompt instructing to write a sonnet with a specified rhyme scheme and include given words.
    Output (y)  : A 14-line sonnet adhering to the specified rhyme scheme and containing all given words verbatim.
    Reward (r)  : 0 or 1, depending on whether the sonnet adheres to the rhyme scheme and includes all given words.

    Input Example:
        Write a sonnet with strict rhyme scheme ABAB CDCD EFEF GG, containing each of the following words verbatim: "grass", "value", and "jail".
    Output Example:
        Line 1: ...
        ...
        Line 14: ...
    """

    def __init__(self, args, file='Sonnets-Standard.jsonl'):
        """
        Initializes the SonnetTask.

        Args:
            file (str): The name of the JSONL file containing sonnet tasks.
        """
        #super().__init__()
        self.__dict__.update(vars(args))
        path = os.path.join(DATA_PATH, 'sonnet', file)
        self.data = self.load_tasks(path)
        self.value_cache = {}
        self.steps = 2  # Sonnet generation typically requires fewer iterative steps
        self.stops = ['\n'] * 14  # Each of the 14 lines ends with a newline
        self.prompts = import_prompts_by_task("sonnet")
        self.free_answer = False
        self.max_turn = 5
        self.expected_action_tokens = 250
        self.threshold = 0.7

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
        return len(self.data)

    def get_input(self, idx: int) -> str:
        """
        Constructs the input prompt for the language model.

        Args:
            idx (int): The index of the current task.

        Returns:
            str: The formatted prompt containing the given words and rhyme scheme.
        """
        task_info = self.data[idx]
        input = task_info['target']


        given_words_str = input.split(',')[-1]

        # Split words into a list
        given_words = [word.strip().lower() for word in given_words_str.split(' ') if word]

        return given_words, None
    def create_root(self, root=None):
        if not root:
            self.root = VerifiedThoughtNode("ROOT", True,[], None, self.performer_model)
        else:
            root.set_config(self.n_extend_layers, not self.none_acc_reward, self.expected_reward_method)
            self.root = root

    def test_output(self, idx: int, output: str):
        """
        Evaluates the generated sonnet for rule adherence and word inclusion.

        Args:
            idx (int): The index of the current task.
            output (str): The generated sonnet.

        Returns:
            dict: {'r': 1} if the sonnet is valid, else {'r': 0}.
        """
        task_info = self.data[idx]
        input_prompt = task_info['input']
        target = task_info['target']

        # Extract words and rhyme scheme from the target
        target_match = re.match(r'([A-Z\s]+),\s*([\w\s]+)', target)
        if not target_match:
            print("Target format is incorrect.")
            return {'r': 0}

        rhyme_scheme = target_match.group(1).replace(" ", "")
        given_words_str = target_match.group(2)
        given_words = [word.strip().lower() for word in given_words_str.split()]

        # Normalize the output for case-insensitive comparison
        output_lower = output.lower()

        # Check if all given words are included verbatim
        words_included = all(word in output_lower for word in given_words)

        # Split the sonnet into lines
        lines = output.strip().split('\n')
        if len(lines) < 14:
            print("Sonnet does not contain 14 lines.")
            return {'r': 0}

        # Extract the last word from each line
        last_words = [self.get_last_word(line) for line in lines[:14]]

        # Map each rhyme scheme letter to its corresponding line indices
        rhyme_map = {}
        for i, rhyming_letter in enumerate(rhyme_scheme):
            rhyme_map.setdefault(rhyming_letter, []).append(i)

        # Check if lines that should rhyme indeed do rhyme
        rhymes_correct = True
        for rhyming_letter, indices in rhyme_map.items():
            if len(indices) < 2:
                continue  # No need to check if only one line has this rhyme

            # Get the rhymes for the first line's last word
            first_word = last_words[indices[0]]
            if not first_word:
                print(f"Line {indices[0]+1} has no last word.")
                rhymes_correct = False
                break

            # Get rhymes using the pronouncing library
            rhymes = pronouncing.rhymes(first_word)
            if not rhymes:
                print(f"No rhymes found for the word '{first_word}'.")
                rhymes_correct = False
                break

            # Verify that all other lines in this rhyme group rhyme with the first word
            for idx_in_rhyme in indices[1:]:
                current_word = last_words[idx_in_rhyme]
                if current_word.lower() not in [w.lower() for w in rhymes] and current_word not in rhymes:
                    print(
                        f"Rhyme mismatch: '{first_word}' does not rhyme with '{current_word}' in lines {indices[0]+1} and {idx_in_rhyme+1}."
                    )
                    rhymes_correct = False
                    break

            if not rhymes_correct:
                break

        # Final evaluation
        if words_included and rhymes_correct:
            return {'r': 1}
        else:
            print("Sonnet failed to include all given words or adhere to the rhyme scheme.")
            return {'r': 0}

    def get_last_word(self, line: str) -> str:
        """
        Extracts the last word from a given line.

        Args:
            line (str): A line of the sonnet.

        Returns:
            str: The last word of the line.
        """
        words = re.findall(r'\b\w+\b', line)
        return words[-1] if words else ''


# Example usage:
# Assuming you have a 'Sonnets-Standard.jsonl' with lines like:
# {"input": "Write a sonnet with strict rhyme scheme ABAB CDCD EFEF GG, containing each of the following words verbatim: \"grass\", \"value\", and \"jail\".", "target": "ABAB CDCD EFEF GG, grass value jail"}

if __name__ == "__main__":
    sonnet_task = SonnetTask(file='Sonnets-Standard.jsonl')
    print(f"Total Sonnet Tasks: {len(sonnet_task)}")
    
    # Example Task
    idx = 0  # Index of the task to process
    given_words = sonnet_task.get_input(idx)
    print("Given words:")
    print(given_words)
    
    # Example Output (replace with actual model-generated sonnet)
    sample_output = (
        "Upon the meadow where the grass does sway,\n"
        "Beneath the stars that shimmer in the night,\n"
        "The silent whispers of the moon's pure light,\n"
        "Reveal the hidden truths that softly play.\n"
        "Each blade that bends reflects a lover's vow,\n"
        "The value of their hearts entwined as one,\n"
        "Through trials faced beneath the setting sun,\n"
        "Their spirits rise above the darkened brow.\n"
        "In gardens where the dreams of youth reside,\n"
        "The echoes of their laughter fill the air,\n"
        "No jail can bind the love they freely bear,\n"
        "As endless as the ocean's endless tide.\n"
        "Thus, bound by rhymes and verses intertwined,\n"
        "Their sonnet speaks of love's immortal bind."
    )
    
    evaluation = sonnet_task.test_output(idx, sample_output)
    print(f"Evaluation: {evaluation}")