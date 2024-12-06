import copy

from chat_utils import renew_open_set
from models import get_response_method
from uop import select, renew_node_to_root
from evaluate_outputs import *


def get_teacher_response(task, history):
	"""
	The Teacher evaluates the Performer's proposed solution for compliance with the provided rules.
	
	Args:
		task: The task configuration containing teacher_model and prompts.
		history: The dialogue history between Performer and Teacher.
	
	Returns:
		Teacher's feedback as a string.
	"""
	response = get_response_method(task.teacher_model)
	# Use the last few messages to keep context relevant
	msg = [history[0]] + history[-3:] if len(history) > 3 else history
	return response(msg, model=task.teacher_model)


def get_performer_response(task, history, ques_id, node):

	"""
	The Performer generates potential solutions based on the given task and rules.
	
	Args:
		task: The task configuration containing performer_model and prompts.
		history: The dialogue history between Performer and Teacher.
		ques_id: The current question/turn identifier.
		node: The current state node in the decision tree.
	
	Returns:
		Tuple containing the updated node, Performer's response, and a flag indicating continuation.
	"""
	print (node.items)
	response = get_response_method(task.performer_model)

	def simplify_rsp(rsp):
		gpt3_response = get_response_method("gpt-3.5-turbo")
		if len(rsp.split(" ")) > task.expected_action_tokens:
			m = [{"role": "user", "content": task.prompts.extract_q_prompt.format(rsp=rsp)}]
			rsp = gpt3_response(m, model="gpt-3.5-turbo", max_tokens=task.expected_action_tokens)
		return rsp

	# If there's only one item to address, prompt accordingly
	if len(node.items) == 1:
		print('****')
		target_request = task.prompts.target_request_FA if task.free_answer else task.prompts.target_request
		formatted_request = target_request.format(target=node.items[0])
		if formatted_request not in [h["content"] for h in history]:
			return node, formatted_request, False
		else:
			targeting_prompt_free = task.prompts.targeting_prompt_free_FA if task.free_answer else task.prompts.targeting_prompt_free
			msg = copy.deepcopy(history) + [{"role": "user", "content": targeting_prompt_free}]
			return node, simplify_rsp(response(msg, model=task.performer_model)), False

	# Decide whether to select a node based on the current question id
	if ques_id < int(task.max_turn * 0.6):
		n = select(task, node)
		if n:
			return n, n.question, True

	# Prompt for a set of items
	targeting_prompt_set = task.prompts.targeting_prompt_set_FA if task.free_answer else task.prompts.targeting_prompt_set
	msg = copy.deepcopy(history) + [{"role": "user", "content": targeting_prompt_set.format(item_list_str=', '.join(node.items))}]
	return node, simplify_rsp(response(msg, model=task.performer_model)), False


def get_performer_naive_response(args, task, input, history, ques_id, lst_wrong):
	"""
	A naive response generator for the Performer without advanced selection logic.
	
	Args:
		task: The task configuration containing performer_model and prompts.
		history: The dialogue history between Performer and Teacher.
		ques_id: The current question/turn identifier.
	
	Returns:
		The Performer's question as a string.
	"""
	response = get_response_method(task.performer_model)
	eval_status = ""

	msg = copy.deepcopy(history)
	prompt = ""
	if 1 < ques_id <= int(task.max_turn * task.threshold):
		prompt += "\nYou can ask the teacher 1 clarifying question about the mistakes you made to better understand. However, the clarifying question must be different from the original question. You cannot ask for hints. The clarifying question must not be a question that can be easily answered by looking at the rules given."


	else:# ques_id > int(task.max_turn * 0.7):
		prompt += task.prompts.urge_prompt 
		if task.inform:
			prompt += task.prompts.inform_prompt.format(list_wrong=', '.join(lst_wrong))
	#prompt += "\nYou can reply me with 1 question to ask only if you're not sure."

	msg[-1]["content"] += " " + prompt

	#print ('------------->', msg)
	rsp = response(msg, model=task.performer_model)

	def extract_question(rsp):
		gpt3_response = get_response_method("gpt-3.5-turbo")
		message = [{"role": "user", "content": task.prompts.extract_q_prompt.format(rsp=rsp)}]
		return gpt3_response(message, model="gpt-3.5-turbo")
	def extract_line(string):
		"""Extracts the entire line of a string where the given regex matches.

		Args:
		string: The input string.
		regex: The regular expression pattern.

		Returns:
		The entire line of the string where the regex matches, or None if no match is found.
		"""
		regex =  r"Answer: (\((.+)\)|=)(.+)= 24"

		match = re.search(regex, string)
		if match:
			start_index = match.start()
			end_index = match.end()
			line_start = string.rfind('\n', 0, start_index)
			line_end = string.find('\n', end_index)
			if line_end == -1:
				line_end = len(string)
			return string[line_start + 1:line_end]
		else:
			return None
	def extract_sonnet(text):
		lines = text.split('\n\n') #[line.strip() for line in text.split('\n') if line.strip()]
		required_structure = [4, 4, 4, 2]
		sonnet = []
		
		for block in lines:
			
			split_block = [line.strip() for line in block.splitlines() if line.strip()]
			#print (split_block, '____\n\n\n***')
			if len(split_block) == 4 or len(split_block) == 2:
				sonnet.append(block)
		
		if sonnet:
			return '\n\n'.join(sonnet)
		return None

	eq = extract_question(rsp)
	"""if 'game24' in args.task:
		if "Answer:" in rsp:
			eq = extract_answer(rsp)
			print ('++++++++++', eq, '++++++++')
			if eq is not None:
				bool_eval = eval_for_GameOf24(input, eq)
				eval_status = "\n Expert verification: the solution is correct and valid." if bool_eval else "\nExpert verification: the solution is not correct."
			else:
				eq = extract_question(rsp)"""
	if 'game24' in args.task:
		eq = extract_line(rsp)
		#print ('++++++++++', eq, '++++++++')
		#print ('**********', rsp, '********')
		if eq is not None:
			bool_eval = eval_for_GameOf24(input, eq.split(':')[-1])
			eval_status = "\n Expert verification: the solution is correct and valid." if bool_eval else "\nExpert verification: the solution is not correct."
			return eq+eval_status

		return rsp+eval_status

	if 'sonnet' in args.task:
		eq = extract_sonnet(rsp)
		if eq is not None:
			bool_eval, error = eval_for_Sonnet(eq)
			eval_status = "\n\nExpert verification: the solution is correct and valid." if bool_eval else "\n\nExpert verification: the solution is not correct." + str(error)
			return eq+eval_status
		return rsp+eval_status

	return eq+eval_status if len(rsp.split(" ")) > task.expected_action_tokens else rsp+eval_status


def converse(args, task, i):
	"""
	Manages the conversation loop between Performer and Teacher until a valid solution is found
	or the maximum number of turns is reached.
	
	Args:
		task: The task configuration containing models, prompts, and data.
		i: The index of the current task instance.
	
	Returns:
		A dictionary containing the conversation history, number of turns, state, and target item.
	"""
	if 'crosswords' in args.task:
		rules = open('rules/crosswords.txt', 'r').read()
		input, gt = (task.get_input(i))
	elif 'game24' in args.task:
		rules = open('rules/game24.txt', 'r').read()
		input, gt = (task.get_input(i))
	elif 'penguin' in args.task:
		rules = open('rules/penguin.txt', 'r').read()
		input, gt = (task.get_input(i))
	elif 'checkmate' in args.task:
		rules = open('rules/checkmate.txt', 'r').read()
		input, gt = (task.get_input(i))
	elif 'sonnet' in args.task:
		rules = open('rules/sonnet.txt', 'r').read()
		input, gt = (task.get_input(i))
	elif 'puzzle' in args.task:
		rules = open('rules/puzzle.txt', 'r').read()
		input, gt = (task.get_input(i))
	
	target_decl = task.prompts.target_declaration.format(target=gt)
	print(target_decl)
	print("------ DIALOGUE START ------")
	count = 0

	# Initialize dialogue history for Teacher
	if not task.free_answer:
		history_e = [{'role': 'user', 'content': task.prompts.examiner_prologue.format(rules=rules, input=input)}]
	else:
		history_e = [{'role': 'user', 'content': task.prompts.performer_prologue.format(rules=rules, conv_hist=task.data[i]["conv_hist"])}]

	# Initialize dialogue history for Performer
	#if "self_repo" in task.data[i]:
	#    performer_prologue = task.prompts.performer_prologue_FA if task.free_answer else task.prompts.performer_prologue
	#    history_g = [{'role': 'user', 'content': performer_prologue.format(repo=task.data[i]["self_repo"])}]
	#    print("Self-report:", task.data[i]["self_repo"])
	#    node = task.root.handle_self_repo(task, task.data[i]["self_repo"])
	#else:
	history_g = [{'role': 'user', 'content': task.prompts.performer_prologue.format(input=input, rules=rules)}]
	# !! for openset verified_thoughts !!
	if task.open_set_size > 0 and task.n_pre_ask > 0:
		for _ in range(task.n_pre_ask):
			bot1_response = get_performer_naive_response(task, history_g, count + 1)
			print("Performer:", bot1_response)
			history_g.append({'role': 'system', 'content': bot1_response})
			history_e.append({'role': 'user', 'content': bot1_response})
			bot2_response = get_teacher_response(task, history_e)
			print("Teacher:", bot2_response)
			history_g.append({'role': 'user', 'content': bot2_response})
			history_e.append({'role': 'system', 'content': bot2_response})
			count += 1
			print('------', count, '-------------')
	node = task.root.handle_self_repo(task, history_g) if task.open_set_size > 0 else task.root

	# Initial Performer response
	node, bot1_response, flag = get_performer_response(task, history_g, count + 1, node)
	print("Performer:", bot1_response)

	history_g.append({'role': 'system', 'content': bot1_response})
	history_e.append({'role': 'user', 'content': bot1_response})

	while True:
		# Teacher evaluates the Performer's response
		bot2_response = get_teacher_response(task, history_e)  # Teacher validation
		if task.free_answer and flag:
			node = node.handle_free_answer(task, bot1_response, bot2_response)
		elif bot2_response.startswith("Valid Solution"):
			node = node.ans2node(True)
		elif bot2_response.startswith("Invalid Solution"):
			node = node.ans2node(False)
		history_g.append({'role': 'user', 'content': bot2_response})
		history_e.append({'role': 'system', 'content': bot2_response})
		print("Teacher:", bot2_response)

		if "Valid Solution" in bot2_response:
			state = 1
			break

		count += 1
		print('------', count, '-------------')

		if count >= task.max_turn:
			print("Teacher: Sorry, time's up. You lose this task.", target_decl)
			state = -1
			break

		# Renew the node if necessary
		if count <= int(task.max_turn * 0.3) + task.n_pre_ask and task.open_set_size > 0 and len(node.items) < task.size_to_renew:
			node = renew_node_to_root(task, node, history_g)

		# Performer generates the next response
		node, bot1_response, flag = get_performer_response(task, history_g, count + 1, node)
		print("Performer:", bot1_response)
		history_g.append({'role': 'system', 'content': bot1_response})
		history_e.append({'role': 'user', 'content': bot1_response})

	if count < task.max_turn:
		state = 1

	return {'turn': count, 'history_g': history_g, 'history_e': history_e, 'state': state, 'gt': gt}


def naive_converse(args, task, i):
	"""
	A simplified conversation manager without advanced node handling.
	
	Args:
		task: The task configuration containing models, prompts, and data.
		i: The index of the current task instance.
	
	Returns:
		A dictionary containing the conversation history, number of turns, state, and target item.
	"""
	if 'crosswords' in args.task:
		rules = open('rules/crosswords.txt', 'r').read()
		input, gt = (task.get_input(i))
	elif 'game24' in args.task:
		rules = open('rules/game24.txt', 'r').read()
		input, gt = (task.get_input(i))
	elif 'penguin' in args.task:
		rules = open('rules/penguin.txt', 'r').read()
		input, gt = (task.get_input(i))
	elif 'checkmate' in args.task:
		rules = open('rules/checkmate.txt', 'r').read()
		input, gt = (task.get_input(i))
	elif 'sonnet' in args.task:
		rules = open('rules/sonnet.txt', 'r').read()
		input, gt = (task.get_input(i))
	elif 'puzzle' in args.task:
		rules = open('rules/puzzle.txt', 'r').read()
		input, gt = (task.get_input(i))

	target_decl = task.prompts.target_declaration.format(target=gt)
	print(target_decl)
	#print ('\nQuestion: ', input.splitlines()[-1])
	print ('\nQuestion: ', input)
	print("------ DIALOGUE START ------")

	#if "self_repo" in task.data[i]:
	#    performer_prologue = task.prompts.performer_prologue_FA if task.free_answer else task.prompts.performer_prologue
	#    history_g = [{'role': 'user', 'content': performer_prologue.format(repo=task.data[i]["self_repo"])}]
	#    print("Self-report:", task.data[i]["self_repo"])
	#else:

	history_e = [{'role': 'user', 'content': task.prompts.examiner_prologue.format(rules=rules, input=input, target=gt)}]

	history_g = [{'role': 'user', 'content': task.prompts.performer_prologue.format(input=input, rules=rules)}]

	count = 0
	lst_wrong = []

	# Initial Performer response
	bot1_response = get_performer_naive_response(args, task, input, history_g, count + 1, lst_wrong)
	print("Performer:", bot1_response)

	history_g.append({'role': 'system', 'content': bot1_response})
	history_e.append({'role': 'user', 'content': bot1_response})

	while True:
		# Teacher evaluates the Performer's response
		bot2_response = get_teacher_response(task, history_e)
		history_g.append({'role': 'user', 'content': bot2_response})
		history_e.append({'role': 'system', 'content': bot2_response})
		print("Teacher:", bot2_response)
		lst_wrong.append(bot1_response)

		if "Valid Solution" in bot2_response:
			state = 1
			break

		count += 1
		
		print('------', count, '-------------')

		if count >= task.max_turn:
			print("Teacher: Sorry, time's up. You lose this task. The correct answer was: ", target_decl)
			state = -1
			break

		# Performer generates the next response
		bot1_response = get_performer_naive_response(args, task, input, history_g, count + 1, lst_wrong)
		lst_wrong.append(bot1_response)
		print("Performer:", bot1_response)
		history_g.append({'role': 'system', 'content': bot1_response})
		history_e.append({'role': 'user', 'content': bot1_response})

	if count < task.max_turn:
		state = 1

	return {'turn': count, 'history_g': history_g, 'history_e': history_e, 'state': state, 'item': gt}