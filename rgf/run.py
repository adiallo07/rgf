import os
import json
import argparse
import pickle

from tqdm import tqdm

from tasks import get_task
from method import converse, naive_converse
from evaluate_outputs import evaluate_performance


def run(args):
    # Initialize the task with the provided arguments
    task = get_task(args)

    # Adjust task indices based on provided arguments
    args.task_start_index = max(args.task_start_index, 0)
    if args.task_end_index < 0:
        args.task_end_index = len(task)
    else:
        args.task_end_index = min(args.task_end_index, len(task.data))

    # Determine the log file path based on whether it's a naive run or not
    if args.naive_run:
        log_file = (
            f'./logs/{args.task}/{args.performer_model}_as_student/{args.dataset}_{args.temperature}'
            f'_naive_{"" if args.inform else "un"}inform_TEACHER_{args.teacher_model}'
            f'_{args.task_start_index}-{args.task_end_index}.json'
        )
    else:
        # Construct the log file path with additional parameters related to the reward system
        log_file = (
            f'./logs/{args.task}/{args.performer_model}_as_student/'
            f'{f"OS_init{args.open_set_size}_renew{args.size_to_renew}_" if args.open_set_size > 0 else ""}'
            f'{f"pre{args.n_pre_ask}_" if args.n_pre_ask > 0 else ""}'
            f'{args.dataset}_{args.temperature}_lambda{args.reward_lambda}_acc{not args.none_acc_reward}'
            f'_exp{args.expected_reward_method}_L{args.n_extend_layers}_K{args.n_potential_actions}'
            f'_PRUN{args.n_pruned_nodes}_TEACHER{args.teacher_model}'
            f'_{args.task_start_index}-{args.task_end_index}.json'
        )

        # Define the root file path for managing the solution tree
        root_file = (
            f'./roots/{args.task}/{args.performer_model}'
            f'{f"OS_init{args.open_set_size}_" if args.open_set_size > 0 else ""}'
            f'_{args.dataset}_{args.temperature}_root.pickle'
        )

        # Load the existing root node if available; otherwise, create a new one
        if os.path.exists(root_file):
            with open(root_file, 'rb') as r:
                root = pickle.load(r)
            task.create_root(root)
        else:
            os.makedirs(os.path.dirname(root_file), exist_ok=True)
            task.create_root()
            with open(root_file, 'wb') as fw:
                pickle.dump(task.root, fw)

    # Ensure the log directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Initialize logs
    logs = []
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            logs = json.load(f)
        args.task_start_index = 1#len(logs)

    # Iterate through the tasks within the specified range
    for i in tqdm(range(args.task_start_index, args.task_end_index), desc="Processing Tasks"):
        
        # For naive runs, use the naive_converse function
        log = naive_converse(args, task, i)
        
        logs.append(log)
        # Save logs incrementally to ensure progress is not lost
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2)

    # After all tasks are processed, evaluate performance
    evaluate_performance(log_file, task)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Verified Thoughts Framework")

    # Model configurations
    parser.add_argument('--performer_model', type=str, default='gpt-4-turbo',
                        choices=['gpt-4', 'gpt-3.5-turbo'])
    parser.add_argument('--temperature', type=float, default=0.)
    parser.add_argument('--teacher_model', type=str, default='gpt-4-turbo')

    # Task configurations
    parser.add_argument('--task', type=str, default='crosswords', choices=['sonnet', 'game24', 'checkmate', 'penguin', 'puzzle', 'crosswords'])
    parser.add_argument('--dataset', type=str, default='common', choices=['sonnet', 'game24', 'checkmate', 'penguin', 'puzzle'])

    # Task indexing
    parser.add_argument('--task_start_index', type=int, default=-1)
    parser.add_argument('--task_end_index', type=int, default=-1)

    # Run modes
    parser.add_argument('--naive_run', action='store_true', default=True, help="Activate naive run mode")


    # Parse and return the arguments
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print("Running with the following configuration:")
    print(args)
    run(args)
