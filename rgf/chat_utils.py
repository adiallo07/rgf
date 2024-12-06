# verified_thoughts/chat_utils.py

import copy
import importlib
import ast

from models import get_response_method

# Mapping task names to their corresponding prompt modules
task_parameter_mapping = {
    "crosswords": "crosswords",
    "checkmate": "checkmate",
    "game24": "game24",
    "sonnet": "sonnet", 
    "penguin": "penguin",
    "puzzle":"puzzle"
}

def import_prompts_by_task(task_name):
    """
    Dynamically imports the prompt module based on the task name.
    
    Args:
        task_name (str): The identifier for the task.
    
    Returns:
        module: The imported prompt module.
    
    Raises:
        ImportError: If the module corresponding to the task name cannot be found.
    """
    parameter = task_parameter_mapping.get(task_name)
    if not parameter:
        raise ValueError(f"Unknown task name: {task_name}")
    module_name = f"prompts.{parameter}"
    try:
        module = importlib.import_module(module_name)
        return module
    except ImportError:
        raise ImportError(f"Failed to import module: {module_name}")

def generate_and_validate_solution(task, items: list, n_solutions: int, asked_solutions: list = None, retry=False):
    """
    Generates potential solutions using the Performer and validates them using the Teacher.
    
    Args:
        task: The task configuration object.
        items (list): Relevant items or information related to the task.
        n_solutions (int): Number of solutions to generate.
        asked_solutions (list, optional): Solutions already proposed to avoid repetition.
        retry (bool, optional): Indicates if this is a retry after a failure.
    
    Returns:
        list: A list of dictionaries containing solutions and their validation status.
    """
    performer = get_response_method(task.performer_model)
    teacher = get_response_method(task.teacher_model)
    solutions = []
    
    for _ in range(n_solutions):
        # Construct the Performer prompt
        if not retry:
            prompt = task.prompts["solution_request"].format(
                items_str=', '.join(items),
                rules='\n'.join([f"{i+1}. {rule}" for i, rule in enumerate(task.rules)]),
                n=n_solutions
            )
        else:
            # Adjust prompt for retries if necessary
            prompt = task.prompts["solution_retry_request"].format(
                items_str=', '.join(items),
                n=n_solutions
            )
        
        messages = [{'role': 'user', 'content': prompt}]
        performer_response = performer(messages)
        performer_solution = performer_response.strip()
        
        # Avoid duplicate solutions
        if performer_solution in (asked_solutions or []):
            continue
        
        # Validate the solution using the Teacher
        validation_feedback = validate_solution(task, performer_solution)
        is_valid = validation_feedback.lower().startswith("valid solution")
        
        solutions.append({
            "solution": performer_solution,
            "is_valid": is_valid,
            "feedback": validation_feedback
        })
    
    return solutions

def validate_solution(task, solution):
    """
    Validates the Performer's solution against the set of rules using the Teacher.
    
    Args:
        task: The task configuration object.
        solution (str): The solution proposed by the Performer.
    
    Returns:
        str: Feedback from the Teacher regarding the solution's validity.
    """
    teacher = get_response_method(task.teacher_model)
    
    # Construct the Teacher prompt
    prompt = task.prompts["teacher_evaluation"].format(
        rules='\n'.join([f"{i+1}. {rule}" for i, rule in enumerate(task.rules)]),
        performer_solution=solution
    )
    
    messages = [{'role': 'user', 'content': prompt}]
    teacher_response = teacher(messages)
    feedback = teacher_response.strip()
    
    return feedback

def initialize_solution_space(task, repo=None):
    """
    Initializes the solution space based on repository information if provided.
    
    Args:
        task: The task configuration object.
        repo (str, optional): Repository or initial information to seed solutions.
    
    Returns:
        list: Initialized solution items.
    """
    performer = get_response_method(task.performer_model)
    teacher = get_response_method(task.teacher_model)
    
    if repo:
        prompt = task.prompts["initialize_solution_space_with_repo"].format(repo=repo)
    else:
        prompt = task.prompts["initialize_solution_space"].format(n=task.initial_solution_space_size)
    
    messages = [{'role': 'user', 'content': prompt}]
    response = performer(messages)
    
    try:
        # Attempt to parse the response as a list
        solution_space = ast.literal_eval(response.strip())
        if isinstance(solution_space, list):
            return solution_space
        else:
            raise ValueError("Parsed solution space is not a list.")
    except (ValueError, SyntaxError):
        # Fallback: Treat the response as comma-separated values
        solution_space = [item.strip() for item in response.strip().split(",") if item.strip()]
        return solution_space

def renew_solution_space(task, history, current_items):
    """
    Renews the solution space based on conversation history and current items.
    
    Args:
        task: The task configuration object.
        history (list): Conversation history.
        current_items (list): Current items or information.
    
    Returns:
        list: Renewed solution space items.
    """
    performer = get_response_method(task.performer_model)
    
    prompt = task.prompts["renew_solution_space"].format(
        history='\n'.join([msg["content"] for msg in history]),
        current_items=', '.join(current_items),
        size=task.open_set_size
    )
    
    messages = copy.deepcopy(history) + [{'role': 'user', 'content': prompt}]
    response = performer(messages)
    
    try:
        # Attempt to parse the response as a list
        renewed_space = ast.literal_eval(response.strip())
        if isinstance(renewed_space, list):
            return renewed_space
        else:
            raise ValueError("Parsed renewed solution space is not a list.")
    except (ValueError, SyntaxError):
        # Fallback: Treat the response as comma-separated values
        renewed_space = [item.strip() for item in response.strip().split(",") if item.strip()]
        return renewed_space
def initialize_open_set(task, repo=""):
    response = get_response_method(task.guesser_model)
    size = task.open_set_size
    
    if isinstance(repo, str):
        message = [{"role": "user", "content": task.prompts.init_open_set_prompt.format(repo=repo, size=size)}]
    else:
        message = repo + [{"role": "user", "content": task.prompts.init_open_set_prompt.format(size=size)}]
    rsp = response(message, model=task.guesser_model, max_tokens=15*size)
    print([rsp])
    try:
        rsp = set(eval(rsp))
        return list(rsp)
    except Exception as e:
        print(e)
        return initialize_open_set(task, repo)


def renew_open_set(task, history, items):
    response = get_response_method(task.guesser_model)
    size = task.open_set_size
    message = copy.deepcopy(history) + [{"role": "user", "content": task.prompts.renew_open_set_prompt.format(size=size, item_list=str(items))}]
    rsp = response(message, model=task.guesser_model, max_tokens=15*size)
    print([rsp])
    try:
        rsp = set(eval(rsp))
        return list(rsp)
    except Exception as e:
        print(e)
        return renew_open_set(task, history, items)

def ques_and_cls_given_items(task, items: list, n, asked_ques: list = None, rest=False):
    response = get_response_method(task.guesser_model)
    if len(items) <= 1:
        return None

    if rest:
        asked = '\n'.join([f"Question {i + 1}: {asked_ques[i]}" for i in range(len(asked_ques))])
        message = [{"role": "user", "content": task.prompts.generate_prompt_rest.format(
            items_str=', '.join(items), n=n, asked=asked, Q1=asked_ques[0])}]
    else:
        asked = "(The question should not be '" + "' or '".join(asked_ques) + "')" if asked_ques else ""
        message = [{"role": "user", "content": task.prompts.generate_prompt.format(items_str=', '.join(items), n=n, asked=asked)}]
    print(message)
    rsp = "#" + response(message, model=task.guesser_model, max_tokens=2000)
    print([rsp])

    def process_ans(rsp):
        ans = []
        for i in range(n):
            if f"Question {i + 1}: " not in rsp:
                continue
            rsp = rsp.split(f"Question {i + 1}: ", 1)[1]
            q = rsp.split("\n", 1)[0]
            rsp = rsp.split("YES: ", 1)[1]
            if rsp[0] == '\n':
                continue
            items_y = rsp.split("\n", 1)[0].split(", ")
            items_y = list(set(items_y))
            rsp = rsp.split("\nNO: ", 1)[1] if "\nNO: " in rsp else rsp.split("NO: ", 1)[1]
            if rsp[0] == '\n':
                continue
            items_n = rsp.split("\n", 1)[0].split(", ")
            items_n = list(set(items_n))
            ans.append({"question": q, "items_yes": items_y, "items_no": items_n})
        return ans

    def format_rsp(rsp):
        gpt3_response = get_response_method("gpt-3.5-turbo")
        message.append({"role": "system", "content": rsp})
        message.append({"role": "user", "content": task.prompts.format_generated_prompt.format(rsp=rsp)})
        return gpt3_response(message, "gpt-3.5-turbo", max_tokens=500)

    try:
        return process_ans(rsp)
    except Exception:
        try:
            rsp = format_rsp(rsp)
            return process_ans(rsp)
        except Exception as e:
            print(e)
            return ques_and_cls_given_items(task, items, n, asked_ques, rest)


def cls_given_repo(task, items: list, repo, translate=False, self_repo=True):
    response = get_response_method(task.guesser_model)
    if self_repo:
        if translate:
            message = [{"role": "user", "content": f"Translate to English: {repo}"}]
            gpt3_response = get_response_method("gpt-3.5-turbo")
            repo = gpt3_response(message, model="gpt-3.5-turbo", max_tokens=500)
        repo = task.prompts.self_repo_prompt.format(repo=repo)
    else:
        repo = task.prompts.free_answer_prompt.format(repo=repo)
    message = [{"role": "user", "content": task.prompts.classify_prompt.format(item_list_str=', '.join(items), repo=repo)}]
    rsp = response(message, model=task.guesser_model, max_tokens=len(items)*(task.expected_target_tokens+5))
    print([rsp])

    def extract_items(rsp, keyword):
        _items = []
        if keyword in rsp:
            rsp_part = rsp.split(keyword, 1)[1]
            if not rsp_part or rsp_part[0] != '\n':
                _items = rsp_part.split("\n", 1)[0].split(", ")
                _items = list(set(_items))
        return _items

    try:
        items_y = extract_items(rsp, "YES: ")
        items_n = extract_items(rsp, "NO: ")
        if len(items_y) == 0 and len(items_n) == 0:
            raise ValueError("No items extracted from the response.")

        return {"items_yes": items_y, "items_no": items_n}

    except Exception as e:
        print(e)
        return cls_given_repo(task, items, repo, translate, self_repo)
