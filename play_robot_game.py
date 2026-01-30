import streamlit as st
from typing import List

def robotSim(commands: List[int], obstacles: List[List[int]]) -> int:
    dirs = [(0,1), (1,0), (0,-1), (-1,0)]
    dir_idx = 0
    x = y = 0
    max_dist_sq = 0
    obstacle_set = set(map(tuple, obstacles))
    for cmd in commands:
        if cmd == -2:
            dir_idx = (dir_idx - 1) % 4
        elif cmd == -1:
            dir_idx = (dir_idx + 1) % 4
        else:
            for _ in range(cmd):
                nx, ny = x + dirs[dir_idx][0], y + dirs[dir_idx][1]
                if (nx, ny) in obstacle_set:
                    break
                x, y = nx, ny
                max_dist_sq = max(max_dist_sq, x*x + y*y)
    return max_dist_sq

st.title('Robot Simulation Game')

commands_str = st.text_input('Enter commands (comma separated, e.g., 4,-1,3):', '4,-1,3')
obstacles_str = st.text_area('Enter obstacles (one per line, e.g., 2,4):', '')

if st.button('Run Simulation'):
    try:
        commands = [int(x.strip()) for x in commands_str.split(',') if x.strip()]
        obstacles = []
        for line in obstacles_str.strip().split('\n'):
            if line.strip():
                parts = [int(x.strip()) for x in line.split(',')]
                if len(parts) == 2:
                    obstacles.append(parts)
        result = robotSim(commands, obstacles)
        st.success(f'Maximum distance squared from origin: {result}')
    except Exception as e:
        st.error(f'Error: {e}')
