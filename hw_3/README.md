# HW3 README  

## Problem 1

### houndify.py File

To use Houndify, the houndify.py file from their SDK should be in the same directory as the notebook. I have committed this file to the repository, so no action should need to be done.

### my_credentials.py Format

**my_credentials.py** should be included as a file. The format of the file is:

```python
# these credentials are used to send mail
GMAIL_USERNAME = "..."
GMAIL_PASSWORD = "..."
GMAIL_SMTP_PORT = 465

# these credentials are used get speech-to-text results
HOUNDIFY_CLIENT_ID = "..."
HOUNDIFY_CLIENT_KEY = "..."
```

### Commands the Monty can do

There are key words that Monty looks for.

#### Sending an email

Say something along the lines of "Hey Monty send an email with subject hello and body goodbye".

Key word is **email**!

Monty hears "email" and knows to send an email. He then searches for "subject" and uses the words following subject. He does the same for "body". If a subject or body is not specified, then Monty uses "(No Subject)" or "(No Body)" respectively. If a subject or body ends in "and", Monty drops the word "and" from the end. (So that, in the example, the subject is "hello", not "hello and")

#### Doing a calculation

Say something along the lines of "Hey Monty calculate three minus 5 plus fifty"

Key word is **calculate**!

Monty takes the words after "calculate" and attempts to solve the equation that they describe. The operations that are supported are addition, subtraction, multiplication, and division. For addition, use the word "plus". For subtraction, use the word "minus". For multiplication, use either "times" or "mutiplied by". For division, use "divided by". Monty skips unrecognized words that are not numbers or the specified operations. If two numbers are said in a row, Monty assumes they are added together. If two operations are said in a row, Monty uses the first one.

#### Telling a joke

Say something along the lines of "Hey Monty tell me a joke".

Key word is **joke**!

Monty hears the word "joke" and requests a joke from [icanhazdadjoke.com](http://icanhazdadjoke.com/). If joke is not said, Monty will not tell a joke. If something like "Monty don't tell me a joke" is said, Monty will still tell a joke. He is simply listening for the word "joke".

## Problem 2

It is assumed that the sound_files/ folder is in the same directory as the homework assignment.




