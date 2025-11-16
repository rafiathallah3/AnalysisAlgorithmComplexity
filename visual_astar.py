import time
import heapq
import random
import matplotlib.pyplot as plt

# ---------- A* Implementation ----------
def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    open_set = []
    heapq.heappush(open_set, (0, start))
    g = {start: 0}
    f = {start: heuristic(start, goal)}
    came_from = {}

    while open_set:
        current = heapq.heappop(open_set)[1]
        if current == goal:
            return True

        x, y = current
        for nx, ny in [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]:
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 0:
                tg = g[current] + 1
                if (nx, ny) not in g or tg < g[(nx, ny)]:
                    g[(nx, ny)] = tg
                    f[(nx, ny)] = tg + heuristic((nx, ny), goal)
                    came_from[(nx, ny)] = current
                    heapq.heappush(open_set, (f[(nx, ny)], (nx, ny)))
    return False

# ---------- Bidirectional A* ----------
def astar_from_side(grid, start, goal, g, f, open_set, closed):
    rows, cols = len(grid), len(grid[0])
    if not open_set:
        return None
    _, current = heapq.heappop(open_set)
    closed.add(current)

    x, y = current
    for nx, ny in [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]:
        if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 0:
            newg = g[current] + 1
            if (nx, ny) not in g or newg < g[(nx, ny)]:
                g[(nx, ny)] = newg
                f[(nx, ny)] = newg + heuristic((nx, ny), goal)
                heapq.heappush(open_set, (f[(nx, ny)], (nx, ny)))
    return current

def bidirectional_astar(grid, start, goal):
    open_f = []
    open_b = []
    heapq.heappush(open_f, (0, start))
    heapq.heappush(open_b, (0, goal))

    g_f, f_f = {start: 0}, {start: heuristic(start, goal)}
    g_b, f_b = {goal: 0}, {goal: heuristic(goal, start)}

    closed_f, closed_b = set(), set()

    while open_f and open_b:
        c1 = astar_from_side(grid, start, goal, g_f, f_f, open_f, closed_f)
        if c1 in closed_b:
            return True

        c2 = astar_from_side(grid, goal, start, g_b, f_b, open_b, closed_b)
        if c2 in closed_f:
            return True

    return False

# ---------- Benchmark ----------
def make_grid(n, obstacle_prob=0.2):
    return [[0 if random.random() > obstacle_prob else 1 for _ in range(n)] for _ in range(n)]

random.seed(2)
regular_time = []
bi_time = []

gridSizes = [10, 25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000]
for i in gridSizes:
    grid_size = i
    grid = make_grid(grid_size)
    start = (0, 0)
    goal = (grid_size-1, grid_size-1)
    grid[0][0] = 0
    grid[-1][-1] = 0

    print("Benchmarking for grid size: ", grid_size)
    # Run benchmarks
    t1 = time.time()
    astar(grid, start, goal)
    t2 = time.time()

    t3 = time.time()
    bidirectional_astar(grid, start, goal)
    t4 = time.time()

    regular_time.append(t2 - t1)
    bi_time.append(t4 - t3)

plt.figure(figsize=(10, 6))
plt.plot(gridSizes, regular_time, 'o-', label="A*")
plt.plot(gridSizes, bi_time, 's-', label="Bidirectional A*")
plt.legend()
plt.xlabel("Grid Size (N x N)")
plt.ylabel("Time (seconds)")
plt.title("Benchmark: Regular A* vs Bidirectional A*")
# Set custom x-axis labels in NxN format
x_labels = [f"{size}x{size}" for size in gridSizes]
plt.xticks(gridSizes, x_labels, rotation=45, ha='right')
plt.tight_layout()
plt.grid(True)
plt.show()