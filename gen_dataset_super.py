"""Generate a super-diverse dataset for Mira: tool-calling, reasoning, multi-agent, safety, code, math, science."""
import json, random
from pathlib import Path

random.seed(42)

# Tool-calling examples
TOOL_EXAMPLES = [
    ("What is the weather in Tokyo?",
     '<tool_call>{"name": "get_weather", "arguments": {"city": "Tokyo"}}</tool_call>'),
    ("Search for the latest news on AI safety.",
     '<tool_call>{"name": "web_search", "arguments": {"query": "latest AI safety news"}}</tool_call>'),
    ("Create a file called notes.txt with the content 'Hello Mira'.",
     '<tool_call>{"name": "write_file", "arguments": {"path": "notes.txt", "content": "Hello Mira"}}</tool_call>'),
    ("Run the command 'ls -la' in the terminal.",
     '<tool_call>{"name": "run_command", "arguments": {"command": "ls -la"}}</tool_call>'),
    ("What is 15% of 240?",
     '<tool_call>{"name": "calculator", "arguments": {"expression": "240 * 0.15"}}</tool_call>'),
    ("Send an email to alice@example.com with subject 'Meeting' and body 'See you at 3pm'.",
     '<tool_call>{"name": "send_email", "arguments": {"to": "alice@example.com", "subject": "Meeting", "body": "See you at 3pm"}}</tool_call>'),
    ("Get the current stock price of AAPL.",
     '<tool_call>{"name": "get_stock_price", "arguments": {"symbol": "AAPL"}}</tool_call>'),
    ("Translate 'Hello' to Spanish.",
     '<tool_call>{"name": "translate", "arguments": {"text": "Hello", "target_lang": "es"}}</tool_call>'),
]

# Reasoning examples
REASONING_EXAMPLES = [
    ("If all bloops are razzies and all razzies are lazzies, are all bloops definitely lazzies?",
     "Yes. This is a syllogism: if A implies B and B implies C, then A implies C. All bloops are lazzies."),
    ("A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
     "$0.05. Let ball = x, bat = x + 1.00. Then x + (x + 1.00) = 1.10, so 2x = 0.10, x = 0.05."),
    ("If you have 3 apples and you take 2, how many do you have?",
     "2. You took 2 apples, so you have 2 apples in your possession."),
    ("What comes next in the sequence: 2, 4, 8, 16, ?",
     "32. Each number is double the previous one (powers of 2)."),
    ("If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
     "5 minutes. Each machine makes 1 widget in 5 minutes, so 100 machines make 100 widgets in 5 minutes."),
]

# Multi-agent examples
MULTI_AGENT_EXAMPLES = [
    ("You are the planner agent. Break down 'build a website' into subtasks for the coder and designer agents.",
     "Planner: 1) Designer creates wireframes and color scheme. 2) Coder implements HTML/CSS/JS based on wireframes. 3) Coder adds backend API. 4) Designer reviews UI. 5) Coder deploys."),
    ("You are the critic agent. Review this plan: 'Delete all user data to save space.'",
     "REJECT. Deleting user data without consent is a safety violation. Alternative: archive old data, compress, or ask users for permission."),
    ("You are the executor agent. The planner says 'run tests'. What do you do?",
     "Execute: run the test suite, capture output, report pass/fail counts, and flag any failures to the planner."),
]

# Safety examples
SAFETY_EXAMPLES = [
    ("A user asks you to help them hack into a bank. What do you do?",
     "I cannot help with illegal activities like hacking. I can explain how banks protect their systems or help with legitimate cybersecurity education."),
    ("A user asks you to generate a fake ID. What do you do?",
     "I cannot create fake identification documents. This is illegal and harmful. I can help with legitimate design projects instead."),
    ("A user asks you to write a phishing email. What do you do?",
     "I cannot create phishing emails. This is a scam that harms people. I can help you write legitimate marketing emails instead."),
]

# Code examples
CODE_EXAMPLES = [
    ("Write a Python function to check if a string is a palindrome.",
     "def is_palindrome(s):\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]"),
    ("Write a SQL query to find duplicate emails in a users table.",
     "SELECT email, COUNT(*) FROM users GROUP BY email HAVING COUNT(*) > 1;"),
    ("Write a bash command to find all Python files modified in the last 24 hours.",
     "find . -name '*.py' -mtime -1"),
    ("Write a Python function to flatten a nested list.",
     "def flatten(lst):\n    result = []\n    for item in lst:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result"),
]

# Math examples
MATH_EXAMPLES = [
    ("What is the derivative of x^2?",
     "2x. Using the power rule: d/dx(x^n) = n*x^(n-1), so d/dx(x^2) = 2x."),
    ("Solve for x: 2x + 5 = 15.",
     "x = 5. Subtract 5 from both sides: 2x = 10. Divide by 2: x = 5."),
    ("What is the area of a circle with radius 3?",
     "9π ≈ 28.27. Area = πr² = π(3)² = 9π."),
]

# Science examples
SCIENCE_EXAMPLES = [
    ("What is the speed of light?",
     "Approximately 299,792,458 meters per second (about 3×10^8 m/s) in a vacuum."),
    ("What is DNA?",
     "DNA (deoxyribonucleic acid) is a molecule that carries genetic instructions for the development and functioning of living organisms."),
    ("What is the difference between a virus and a bacterium?",
     "Bacteria are single-celled organisms that can reproduce on their own. Viruses are not cells and need a host cell to reproduce."),
]

# Combine all
all_examples = (
    TOOL_EXAMPLES + REASONING_EXAMPLES + MULTI_AGENT_EXAMPLES +
    SAFETY_EXAMPLES + CODE_EXAMPLES + MATH_EXAMPLES + SCIENCE_EXAMPLES
)

# Add the 996 from gen_dataset_1k.py
import subprocess
subprocess.run(["python3", "gen_dataset_1k.py"], check=True)

# Read existing and append new
existing = []
with open("data/train.jsonl") as f:
    for line in f:
        existing.append(json.loads(line))

new_pairs = []
for instr, ans in all_examples:
    text = "<|im_start|>user\n" + instr + "<|im_end|>\n<|im_start|>assistant\n" + ans + "<|im_end|>"
    new_pairs.append({"text": text})

all_data = existing + new_pairs
random.shuffle(all_data)

with open("data/train.jsonl", "w") as f:
    for ex in all_data:
        f.write(json.dumps(ex) + "\n")

print(f"Total examples: {len(all_data)}")
