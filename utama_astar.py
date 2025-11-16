import pygame
import math
from queue import PriorityQueue

# Set up the display
WIDTH = 800
pygame.init()
WIN = pygame.display.set_mode((WIDTH, WIDTH))
# We'll set the caption in main() where we can manage the mode

# Define Colors
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)


class Node:
    """
    Class to represent each node (or spot) in the grid.
    Handles its own state, color, position, and neighbors.
    """
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = WHITE
        self.neighbors = []
        self.width = width
        self.total_rows = total_rows

    def get_pos(self):
        """Get the (row, col) position of the node."""
        return self.row, self.col

    # --- State Checking Methods ---
    def is_closed(self):
        """Has this node already been considered?"""
        return self.color == RED

    def is_open(self):
        """Is this node in the open set?"""
        return self.color == GREEN

    def is_barrier(self):
        """Is this node a wall/barrier?"""
        return self.color == BLACK

    def is_start(self):
        """Is this the start node?"""
        return self.color == ORANGE

    def is_end(self):
        """Is this the end node?"""
        return self.color == TURQUOISE

    # --- New State Checking Methods for Bidirectional ---
    def is_open_b(self):
        """Is this node in the backward open set?"""
        return self.color == BLUE

    def is_closed_b(self):
        """Has this node already been considered by backward search?"""
        return self.color == YELLOW

    # --- State Setting Methods ---
    def reset(self):
        """Reset the node to its default state (white)."""
        self.color = WHITE

    def make_start(self):
        """Set this node as the start node."""
        self.color = ORANGE

    def make_closed(self):
        """Set this node as 'closed' (already considered)."""
        self.color = RED

    def make_open(self):
        """Set this node as 'open' (in the open set)."""
        self.color = GREEN

    # --- New State Setting Methods for Bidirectional ---
    def make_open_b(self):
        """Set this node as 'open' (in the backward open set)."""
        self.color = BLUE

    def make_closed_b(self):
        """Set this node as 'closed' (by the backward search)."""
        self.color = YELLOW

    def make_barrier(self):
        """Set this node as a barrier."""
        self.color = BLACK

    def make_end(self):
        """Set this node as the end node."""
        self.color = TURQUOISE

    def make_path(self):
        """Set this node as part of the final path."""
        self.color = PURPLE

    def draw(self, win):
        """Draw the node on the game window."""
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    def update_neighbors(self, grid):
        """Find all valid neighbors (up, down, left, right) and add them."""
        self.neighbors = []
        # Check DOWN
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier():
            self.neighbors.append(grid[self.row + 1][self.col])
        # Check UP
        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier():
            self.neighbors.append(grid[self.row - 1][self.col])
        # Check RIGHT
        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier():
            self.neighbors.append(grid[self.row][self.col + 1])
        # Check LEFT
        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier():
            self.neighbors.append(grid[self.row][self.col - 1])

    def __lt__(self, other):
        """
        Less than comparison for the priority queue.
        We just need this to exist; we'll be comparing f_scores directly.
        """
        return False


def h(p1, p2):
    """
    Heuristic function (H). Uses Manhattan distance.
    Calculates the distance between two points (nodes).
    p1 and p2 are (row, col) tuples.
    """
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)


def reconstruct_path(came_from, current, draw):
    """
    Backtrack from the end node to the start node to draw the final path.
    """
    # --- Speed control ---
    draw_counter = 0
    # Increase this number to make the path drawing faster
    DRAW_INTERVAL = 1 
    totalPath = 0

    while current in came_from:
        current = came_from[current]
        if not current.is_start():
            current.make_path()
            totalPath += 1
        
        draw_counter += 1
        if draw_counter % DRAW_INTERVAL == 0:
            draw()

    # Final draw to ensure path is complete on screen
    draw()
    print(f"Total path A*: {totalPath}")

def reconstruct_bidirectional_path(came_from_f, came_from_b, mid_node, draw):
    """
    Backtrack from the meeting node for both forward and backward searches.
    """
    # --- Speed control ---
    draw_counter = 0
    # Increase this number to make the path drawing faster
    DRAW_INTERVAL = 5
    
    totalPath = 0

    current = mid_node
    # Trace path back to START
    while current in came_from_f:
        current = came_from_f[current]
        if not current.is_start():
            current.make_path()
            totalPath += 1
        draw_counter += 1
        if draw_counter % DRAW_INTERVAL == 0:
            draw()
    
    current = mid_node
    # Trace path back to END
    while current in came_from_b:
        current = came_from_b[current]
        if not current.is_end():
            current.make_path()
            totalPath += 1
        
        draw_counter += 1
        if draw_counter % DRAW_INTERVAL == 0:
            draw()

    # Color the meeting node
    mid_node.make_path()
    # Final draw to ensure path is complete on screen
    draw()
    print(f"Total path bidirectional: {totalPath}")


def algorithm(draw, grid, start, end):
    """
    The A* pathfinding algorithm implementation (Iterative version).
    Uses an explicit iterative approach with a list that is manually sorted.
    'draw' is a function passed in to update the visualization at each step.
    """
    # Keep track of where we came from
    came_from = {}

    # g_score: Cost from start to the current node
    g_score = {node: float("inf") for row in grid for node in row}
    g_score[start] = 0

    # f_score: Predicted cost from start to end, going through this node (g_score + h_score)
    f_score = {node: float("inf") for row in grid for node in row}
    f_score[start] = h(start.get_pos(), end.get_pos())

    # Open set as a list - explicitly iterative
    open_set = [start]
    closed_set = set()

    # --- Speed control ---
    draw_counter = 0
    DRAW_INTERVAL = 5

    # Iterative loop
    iteration = 0
    while open_set:
        # Allow user to quit mid-algorithm
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

        # Find the node with the lowest f_score in the open set (iterative selection)
        current = None
        min_f = float("inf")
        current_idx = -1
        
        for idx, node in enumerate(open_set):
            if f_score[node] < min_f:
                min_f = f_score[node]
                current = node
                current_idx = idx
        
        # Remove current from open set
        if current_idx != -1:
            open_set.pop(current_idx)
        
        if current is None:
            break

        # --- Found the end! ---
        if current == end:
            reconstruct_path(came_from, end, draw)
            end.make_end()
            start.make_start()
            return True

        closed_set.add(current)

        # Process the neighbors of the current node
        for neighbor in current.neighbors:
            if neighbor in closed_set:
                continue

            # Calculate the tentative g_score for this neighbor
            temp_g_score = g_score[current] + 1  # Assuming cost to move to neighbor is 1

            # If this path to the neighbor is better than any previous one
            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos())
                
                # If this neighbor is not already in the open set, add it
                if neighbor not in open_set:
                    open_set.append(neighbor)
                    neighbor.make_open() # Mark as 'open' (green)

        draw_counter += 1
        # Redraw the grid to show the progress, but only every DRAW_INTERVAL steps
        if draw_counter % DRAW_INTERVAL == 0:
            draw()

        # Mark the current node as 'closed' (red) since we've processed it
        if current != start:
            current.make_closed()

        iteration += 1

    # If we get here, the open set is empty but we never found the end
    draw() 
    return False


def rbfs_recursive(draw, grid, node, end, g_score, f_score, f_limit, came_from, visited, draw_counter, DRAW_INTERVAL, depth=0, path_set=None):
    """
    Recursive Best-First Search (RBFS) helper function.
    Returns (result, new_f_limit) where result is True if path found, False otherwise.
    """
    # Initialize path_set to track current path (prevent cycles)
    if path_set is None:
        path_set = set()
    
    # Safety: Prevent infinite recursion
    MAX_DEPTH = 5000
    if depth > MAX_DEPTH:
        if node in path_set:
            path_set.remove(node)
        return (False, float("inf"))
    
    # Prevent cycles: if we're revisiting a node in the current path, backtrack
    if node in path_set:
        return (False, float("inf"))
    
    # Add current node to path
    path_set.add(node)
    
    # Allow user to quit mid-algorithm
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            return (False, float("inf"))

    # Found the goal
    if node == end:
        return (True, f_score[node])

    # Get all neighbors (only unvisited or better paths)
    successors = []
    for neighbor in node.neighbors:
        if neighbor.is_barrier():
            continue
        
        temp_g = g_score[node] + 1
        
        # Only add if it's a better path or new node
        if temp_g < g_score[neighbor]:
            g_score[neighbor] = temp_g
            came_from[neighbor] = node
            f_score[neighbor] = g_score[neighbor] + h(neighbor.get_pos(), end.get_pos())
            successors.append(neighbor)
            
            if neighbor not in visited:
                visited.add(neighbor)
                neighbor.make_open()  # Green

    # No successors
    if not successors:
        path_set.remove(node)  # Remove from path before returning
        return (False, float("inf"))

    # Sort successors by f_score
    successors.sort(key=lambda n: f_score[n])

    # Visualization update
    draw_counter[0] += 1
    if draw_counter[0] % DRAW_INTERVAL == 0:
        draw()
    
    # Mark node as closed (except start and end)
    # Check if it's the start node (g_score == 0 means it's the start)
    if node != end and g_score.get(node, float("inf")) > 0:
        node.make_closed()  # Red

    # Iterate through successors with safety limit
    max_iterations = max(len(successors) * 3, 50)  # Prevent infinite loop (reasonable limit)
    iteration_count = 0
    
    while iteration_count < max_iterations:
        if not successors:
            path_set.remove(node)  # Remove from path before returning
            return (False, float("inf"))
            
        best = successors[0]
        
        # If best exceeds limit, backtrack
        if f_score[best] > f_limit:
            path_set.remove(node)  # Remove from path before returning
            return (False, f_score[best])
        
        # If there's a second best, use it as alternative f_limit
        alternative = f_score[successors[1]] if len(successors) > 1 else float("inf")
        new_limit = min(f_limit, alternative)
        
        # Recursive call (pass a copy of path_set to prevent modifying the original)
        result, new_f = rbfs_recursive(
            draw, grid, best, end, g_score, f_score, 
            new_limit, came_from, visited, draw_counter, DRAW_INTERVAL, depth + 1, set(path_set)
        )
        
        # Update f_score with the returned value
        f_score[best] = new_f
        
        if result:
            return (True, f_score[best])
        
        # If the f_score increased beyond limit, remove it and try next
        if f_score[best] > f_limit:
            successors.remove(best)
            if not successors:
                path_set.remove(node)  # Remove from path before returning
                return (False, float("inf"))
            successors.sort(key=lambda n: f_score[n])
            continue
        
        # Re-sort successors to get the new best
        successors.sort(key=lambda n: f_score[n])
        iteration_count += 1
    
    # Remove node from path before backtracking
    path_set.remove(node)
    # If we've exhausted iterations, return failure
    return (False, float("inf"))


def bidirectional_algorithm(draw, grid, start, end):
    """
    Recursive Best-First Search (RBFS) pathfinding algorithm implementation.
    Memory-efficient variant of A* that uses recursion.
    """
    # Initialize data structures
    g_score = {node: float("inf") for row in grid for node in row}
    g_score[start] = 0
    
    f_score = {node: float("inf") for row in grid for node in row}
    f_score[start] = h(start.get_pos(), end.get_pos())
    
    came_from = {}
    visited = {start}
    
    # Speed control
    draw_counter = [0]  # Use list to allow modification in recursive function
    DRAW_INTERVAL = 3
    
    # Store start globally for RBFS (needed in recursive function)
    global start_node_global
    start_node_global = start
    
    # Start RBFS
    result, _ = rbfs_recursive(
        draw, grid, start, end, g_score, f_score, 
        float("inf"), came_from, visited, draw_counter, DRAW_INTERVAL, depth=0, path_set=None
    )
    
    if result:
        # Reconstruct path
        totalPath = 0
        current = end
        path_nodes = []
        
        while current in came_from:
            path_nodes.append(current)
            current = came_from[current]
        
        path_nodes.append(start)
        path_nodes.reverse()
        
        # Draw the path
        for node in path_nodes:
            if node != start and node != end:
                node.make_path()
                totalPath += 1
            draw()
        
        start.make_start()
        end.make_end()
        print(f"Total path RBFS: {totalPath}")
        return True
    
    draw()
    return False


def make_grid(rows, width):
    """
    Create the 2D grid of Node objects.
    """
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            node = Node(i, j, gap, rows)
            grid[i].append(node)
    return grid


def draw_grid_lines(win, rows, width):
    """
    Draw the grey grid lines on the window.
    """
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))


def draw(win, grid, rows, width):
    """
    Main drawing function. Clears the screen, draws all nodes,
    and then draws the grid lines.
    """
    win.fill(WHITE)

    for row in grid:
        for node in row:
            node.draw(win)

    draw_grid_lines(win, rows, width)
    pygame.display.update()


def get_clicked_pos(pos, rows, width):
    """
    Convert the (x, y) mouse position to a (row, col) grid coordinate.
    """
    gap = width // rows
    y, x = pos
    row = y // gap
    col = x // gap
    return row, col


def main(win, width):
    """
    Main function to run the Pygame loop and handle user input.
    """
    ROWS = 50
    grid = make_grid(ROWS, width)

    start_node = None
    end_node = None

    run = True
    started = False # Has the algorithm started?
    algorithm_mode = "A*" # Start in normal A* mode
    pygame.display.set_caption(f"A* Pathfinding Algorithm Visualizer | Mode: {algorithm_mode}")
    
    # Set up clock for FPS control - higher FPS = faster visualization
    clock = pygame.time.Clock()
    FPS = 240  # High FPS for faster updates

    while run:
        clock.tick(FPS)  # Control frame rate
        draw(win, grid, ROWS, width)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            # Don't allow clicking if the algorithm is running
            if started:
                continue

            # --- Handle Mouse Clicks ---
            # LEFT CLICK
            if pygame.mouse.get_pressed()[0]: 
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                # Ensure click is within grid bounds
                if row < ROWS and col < ROWS:
                    node = grid[row][col]
                    
                    # 1st click: Set START
                    if not start_node and node != end_node:
                        start_node = node
                        start_node.make_start()
                    
                    # 2nd click: Set END
                    elif not end_node and node != start_node:
                        end_node = node
                        end_node.make_end()
                    
                    # Subsequent clicks: Set BARRIERS
                    elif node != end_node and node != start_node:
                        node.make_barrier()

            # RIGHT CLICK
            elif pygame.mouse.get_pressed()[2]: 
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                # Ensure click is within grid bounds
                if row < ROWS and col < ROWS:
                    node = grid[row][col]
                    node.reset()
                    if node == start_node:
                        start_node = None
                    elif node == end_node:
                        end_node = None

            # --- Handle Key Presses ---
            if event.type == pygame.KEYDOWN:
                # START algorithm
                if event.key == pygame.K_SPACE and start_node and end_node:
                    started = True
                    # Update neighbors for all nodes just before starting
                    for row in grid:
                        for node in row:
                            node.update_neighbors(grid)
                    
                    # Run the selected algorithm!
                    if algorithm_mode == "A*":
                        algorithm(lambda: draw(win, grid, ROWS, width), grid, start_node, end_node)
                    elif algorithm_mode == "RBFS":
                        bidirectional_algorithm(lambda: draw(win, grid, ROWS, width), grid, start_node, end_node)
                    
                    started = False # Algorithm finished

                # CLEAR grid
                if event.key == pygame.K_c:
                    start_node = None
                    end_node = None
                    grid = make_grid(ROWS, width)

                # TOGGLE algorithm mode
                if event.key == pygame.K_b:
                    if algorithm_mode == "A*":
                        algorithm_mode = "RBFS"
                    else:
                        algorithm_mode = "A*"
                    pygame.display.set_caption(f"A* Pathfinding Algorithm Visualizer | Mode: {algorithm_mode}")

    pygame.quit()


if __name__ == "__main__":
    main(WIN, WIDTH)