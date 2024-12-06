from prompts.general import *



# Performer (Generator) Prompts
performer_prologue = '''You're an expert poet. Let us compose a sonnet adhering strictly to the specified rhyme scheme and containing each of the following words verbatim {input}.

Here are the rules you must follow:
{rules}

You will write a 14-line sonnet that follows the rhyme scheme ABAB CDCD EFEF GG. Ensure that all the provided words are seamlessly integrated. You can choose the theme and the emotion.

Let us begin. Please compose your first sonnet incorporating the provided words.
'''

urge_prompt = """Note that you should write what the solution is from now on.
The prompt must start with 'The answer is ..."""  # Optionally, can be customized or kept empty based on specific needs


# Teacher (Reviewer) Prompts
examiner_prologue = '''Let us review the sonnet that should contain each of the following words verbatim {input}, to ensure it adheres to the specified guidelines here:
{rules}

You are the Examiner responsible for verifying the Performer's proposed sonnet.
An expert may help you verify the performer response and will give a statement starting with "Expert verification: ..."

If the sonnet is correct, respond with: "Valid Solution", otherwise "Invalid Solution".

If there are any discrepancies or errors, specify which rules are incorrect and provide constructive feedback for correction but do not rewrite the sonnet. Point at the lines containing the mistakes.

Let us begin. Here is the proposed sonnet:
'''

