import sys
sys.path.insert(0, 'D:/repo/lixia_homejob/llm-rec-interest-qwen')
from agent.gmail_tool import resolve_recipient
print("Test 1:", resolve_recipient('Lin Yang <lin.yang@example.com>'))
print("Test 2:", resolve_recipient('lin yang'))
print("Test 3:", resolve_recipient('lynn69688@gmail.com'))
