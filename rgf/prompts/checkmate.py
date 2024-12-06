from prompts.general import *


# Conversation Prompts
target_question = "Does the move '{target}' result in a checkmate?"

# Performer (Guesser) Prompts
performer_prologue = '''Let us play chess. You are an expert chess player who will analyze the provided chess moves and determine the next move that will result in a checkmate.

You will suggest a single chess move that, when played, will checkmate the opponent. The move must result in an immediate checkmate.
Take into account the Teacher's feedback and do not repeat wrong moves.

Previous moves:
{input}

First, write the position of all pieces on the board based on the previous moves.


The rules you must follow are:
{rules}

Let us begin. Please propose your next move or ask a clarifying question.
'''

inform_prompt = """Here are the wrong solutions previously suggested: {list_wrong}. Do not suggest them again."""
urge_prompt = """Note that you should guess what the answer is exactly from now on without repeating your mistakes.
The answer must start with 'The answer is ..."""  # Optionally, can be customized or kept empty based on specific needs


# Teacher (Examiner) Prompts
examiner_prologue = '''Let us review the proposed move to ensure it results in a checkmate.
Here are the previous moves:
{input}

You are the Examiner responsible for verifying the Performer's suggested move and compare it to the correct move.

Here is the correct move: {target}. Do not mention it in your dialog with the Performer.

Please check the following rules:
{rules}

If the move is correct and results in a checkmate, respond with: "Valid Solution" otherwise "Invalid Solution".

If the move does not result in a checkmate or violates any rules, specify why it is incorrect, which rule it violates and provide constructive feedback for guidance but do not suggest the correct move.

Let us begin. Here is the proposed move by the Performer:

'''
