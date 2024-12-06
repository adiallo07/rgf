from prompts.general import *


# performer (guesser)
performer_prologue = '''Let us engage in an Animal QA session. The goal is to answer questions about a table of penguins and their attributes. You are the Performer who will propose potential answers based on the provided tables containing information about various animals.
Here is the table information:
{input}

You will suggest one possible answer at a time based on the question. Each answer should be directly supported by the data in the tables.

You must follow these rules:
{rules}

Let us begin. Please provide your first answer to the question. Write the solution in this following format "The answer is ... "  
'''

urge_prompt = """Note that you should guess what the answer exactly is from now on.
The prompt must start with 'The answer is ...'"""

# teacher (examiner)
examiner_prologue = '''Let us engage in an Animal QA session. You are the Teacher responsible for verifying the Performer's proposed answers.
{input}
Please verify the following:
{rules}

Compare the Performer answer with the ground truth answer {target}.
If the Performer's answer is correct, respond with: "Valid Solution" otherwise respond with "Invalid Solution".

If there are any discrepancies, specify which rules are incorrect and provide constructive feedback for correction but the correct answer must not be given in your answer. If the performer asks a clarifying question, reply to the question but do not give the correct answer and rewrite the original question.
Let us begin. 
'''
