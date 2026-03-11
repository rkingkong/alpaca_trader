#!/usr/bin/env python3
"""Fix indentation in weighted consensus block (line ~1214)."""
import os

target = os.path.join(os.path.dirname(os.path.abspath(__file__)), "01_market_analysis.py")

with open(target, "r", encoding="utf-8") as f:
    lines = f.readlines()

fixed = 0
for i, line in enumerate(lines):
    # Find the over-indented lines (36 spaces instead of 16)
    if "_weights = self.feedback_context.get" in line and line.startswith("                                    "):
        lines[i] = "                " + line.lstrip()
        fixed += 1
    elif "_w_claude = _weights.get" in line and line.startswith("                                    "):
        lines[i] = "                " + line.lstrip()
        fixed += 1
    elif "_w_gpt = _weights.get" in line and line.startswith("                                    "):
        lines[i] = "                " + line.lstrip()
        fixed += 1
    elif "_bonus = _weights.get" in line and line.startswith("                                    "):
        lines[i] = "                " + line.lstrip()
        fixed += 1
    elif "consensus_confidence = min(1.0, (claude_conf" in line and line.startswith("                                    "):
        lines[i] = "                " + line.lstrip()
        fixed += 1

with open(target, "w", encoding="utf-8") as f:
    f.writelines(lines)

print(f"Fixed {fixed} lines")

# Verify syntax
import ast
with open(target, "r", encoding="utf-8") as f:
    source = f.read()
try:
    ast.parse(source)
    print("Syntax OK!")
except SyntaxError as e:
    print(f"Still broken at line {e.lineno}: {e.msg}")