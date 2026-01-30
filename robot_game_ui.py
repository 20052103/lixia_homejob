import tkinter as tk
from tkinter import ttk, messagebox
from typing import List

# --- Robot logic ---
def robotSim(commands: List[int], obstacles: List[List[int]], visualize_callback=None) -> int:
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
                if visualize_callback:
                    visualize_callback(x, y, dir_idx)
    return max_dist_sq

# --- Tkinter UI ---
CELL_SIZE = 40
GRID_RANGE = 15  # -15 to 15 in each direction
CANVAS_SIZE = CELL_SIZE * (GRID_RANGE * 2 + 1)

class RobotGameUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Robot Simulation Game")
        
        # Input frame
        input_frame = ttk.Frame(root)
        input_frame.pack(padx=10, pady=10, fill=tk.X)
        
        ttk.Label(input_frame, text="Commands (comma separated):").pack(side=tk.LEFT, padx=5)
        self.commands_entry = ttk.Entry(input_frame, width=30)
        self.commands_entry.insert(0, "4,-1,4,-2,4")
        self.commands_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(input_frame, text="Load Example 1", command=self.load_example1).pack(side=tk.LEFT, padx=2)
        ttk.Button(input_frame, text="Load Example 2", command=self.load_example2).pack(side=tk.LEFT, padx=2)
        ttk.Button(input_frame, text="Load Example 3", command=self.load_example3).pack(side=tk.LEFT, padx=2)
        
        # Obstacles frame
        obs_frame = ttk.Frame(root)
        obs_frame.pack(padx=10, pady=10, fill=tk.X)
        
        ttk.Label(obs_frame, text="Obstacles (x,y per line):").pack(side=tk.LEFT, padx=5)
        self.obstacles_entry = tk.Text(obs_frame, height=3, width=30)
        self.obstacles_entry.insert("1.0", "2,4")
        self.obstacles_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(root, text="Run Simulation", command=self.run_simulation).pack(pady=10)
        
        # Canvas
        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="white", highlightthickness=1, highlightbackground="black")
        self.canvas.pack(padx=10, pady=10)
        
        # Result label
        self.result_label = ttk.Label(root, text="", font=("Arial", 12, "bold"))
        self.result_label.pack(pady=10)
        
        # Initialize
        self.path = [(0, 0)]
        self.robot_pos = (0, 0)
        self.robot_dir = 0
        self.obstacles = []
        self.draw_canvas()

    def load_example1(self):
        self.commands_entry.delete(0, tk.END)
        self.commands_entry.insert(0, "4,-1,3")
        self.obstacles_entry.delete("1.0", tk.END)
        self.obstacles_entry.insert("1.0", "")

    def load_example2(self):
        self.commands_entry.delete(0, tk.END)
        self.commands_entry.insert(0, "4,-1,4,-2,4")
        self.obstacles_entry.delete("1.0", tk.END)
        self.obstacles_entry.insert("1.0", "2,4")

    def load_example3(self):
        self.commands_entry.delete(0, tk.END)
        self.commands_entry.insert(0, "6,-1,-1,6")
        self.obstacles_entry.delete("1.0", tk.END)
        self.obstacles_entry.insert("1.0", "0,0")

    def to_canvas_coords(self, x, y):
        cx = GRID_RANGE * CELL_SIZE + x * CELL_SIZE
        cy = GRID_RANGE * CELL_SIZE - y * CELL_SIZE
        return cx, cy

    def draw_canvas(self):
        self.canvas.delete("all")
        
        # Draw grid
        for i in range(-GRID_RANGE, GRID_RANGE + 1):
            x = GRID_RANGE * CELL_SIZE + i * CELL_SIZE
            self.canvas.create_line(x, 0, x, CANVAS_SIZE, fill="lightgray")
            y = GRID_RANGE * CELL_SIZE - i * CELL_SIZE
            self.canvas.create_line(0, y, CANVAS_SIZE, y, fill="lightgray")
        
        # Draw origin
        ox, oy = self.to_canvas_coords(0, 0)
        self.canvas.create_oval(ox - 3, oy - 3, ox + 3, oy + 3, fill="black")
        
        # Draw obstacles
        for ox, oy in self.obstacles:
            cx, cy = self.to_canvas_coords(ox, oy)
            self.canvas.create_rectangle(cx - CELL_SIZE//2, cy - CELL_SIZE//2, cx + CELL_SIZE//2, cy + CELL_SIZE//2, fill="red")
        
        # Draw path
        for px, py in self.path:
            cx, cy = self.to_canvas_coords(px, py)
            self.canvas.create_oval(cx - 5, cy - 5, cx + 5, cy + 5, fill="lightblue")
        
        # Draw robot
        rx, ry = self.to_canvas_coords(*self.robot_pos)
        self.canvas.create_oval(rx - 8, ry - 8, rx + 8, ry + 8, fill="blue")
        
        # Draw direction arrow
        directions = [(0, -15), (15, 0), (0, 15), (-15, 0)]
        dx, dy = directions[self.robot_dir]
        self.canvas.create_line(rx, ry, rx + dx, ry + dy, fill="black", width=2)

    def run_simulation(self):
        try:
            commands_str = self.commands_entry.get()
            commands = [int(x.strip()) for x in commands_str.split(',') if x.strip()]
            
            obstacles_text = self.obstacles_entry.get("1.0", tk.END).strip()
            obstacles = []
            for line in obstacles_text.split('\n'):
                if line.strip():
                    parts = [int(x.strip()) for x in line.split(',')]
                    if len(parts) == 2:
                        obstacles.append(parts)
            
            self.path = [(0, 0)]
            self.robot_pos = (0, 0)
            self.robot_dir = 0
            self.obstacles = obstacles
            self.draw_canvas()
            self.root.update()
            
            def visualize_callback(x, y, dir_idx):
                self.robot_pos = (x, y)
                self.robot_dir = dir_idx
                self.path.append((x, y))
                self.draw_canvas()
                self.root.update()
                self.root.after(300)
            
            result = robotSim(commands, obstacles, visualize_callback)
            self.result_label.config(text=f"Max distance squared: {result}", foreground="green")
            messagebox.showinfo("Success", f"Simulation complete!\nMax distance squared: {result}")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input:\n{e}")
            self.result_label.config(text="Error", foreground="red")

def main():
    root = tk.Tk()
    app = RobotGameUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()
