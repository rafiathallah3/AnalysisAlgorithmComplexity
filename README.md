# Exponentiation Algorithm Complexity Visualization

This project visualizes and compares the time complexity of three different algorithms for calculating exponentiation (x^n). The visualization demonstrates how different algorithmic approaches perform as the exponent size increases.

## Algorithms Implemented

### 1. Iterative Naive Algorithm
- **Function**: `pangkat_iterasi_naif(x, n)`
- **Approach**: Uses a simple loop to multiply the base `x` by itself `n` times
- **Time Complexity**: O(n)
- **Space Complexity**: O(1)

### 2. Recursive Naive Algorithm
- **Function**: `pangkat_rekursif_naif(x, n)`
- **Approach**: Recursively multiplies `x` by the result of `x^(n-1)` until reaching the base case
- **Time Complexity**: O(n)
- **Space Complexity**: O(n) due to recursion stack

### 3. Recursive Divide and Conquer Algorithm (Fast Exponentiation)
- **Function**: `pangkat_rekursif_cepat(x, n)`
- **Approach**: Uses the "exponentiation by squaring" technique:
  - If `n` is even: `x^n = (x^(n/2))^2`
  - If `n` is odd: `x^n = x * (x^((n-1)/2))^2`
- **Time Complexity**: O(log n)
- **Space Complexity**: O(log n) due to recursion stack

## Requirements

- Python 3.x
- matplotlib

## Installation

1. Make sure you have Python 3.x installed on your system.

2. Install the required dependency:
   ```bash
   pip install matplotlib
   ```

   Or if you're using Python 3 specifically:
   ```bash
   pip3 install matplotlib
   ```

## How to Run

1. Navigate to the project directory:
   ```bash
   cd AnalysisAlgorithmComplexityProject
   ```

2. Run the visualization script:
   ```bash
   python visual_exponent.py
   ```
   
   Or on some systems:
   ```bash
   python3 visual_exponent.py
   ```

3. The script will:
   - Calculate 2^n for values of n from 1 to 10,000 (in steps of 20)
   - Measure the execution time for each algorithm
   - Display a graph comparing the performance of all three algorithms

4. A matplotlib window will open showing the performance comparison graph. The graph displays:
   - X-axis: Input size `n` (the exponent)
   - Y-axis: Execution time in seconds
   - Three lines representing each algorithm's performance

## Expected Results

The visualization will clearly show:
- **Iterative Naive** and **Recursive Naive** algorithms have similar linear O(n) performance, with execution time increasing proportionally with the exponent size
- **Recursive Divide and Conquer** algorithm demonstrates logarithmic O(log n) performance, showing significantly better scalability as the exponent size increases
- The divide and conquer approach becomes increasingly more efficient compared to the naive methods as `n` grows larger

## Notes

- The script sets the recursion limit to 10,000 to handle large recursive calls
- The base number is set to 2 by default (can be modified in the code)
- The exponent range tested is from 1 to 10,000 with increments of 20
- Execution times are measured using `time.perf_counter()` for high precision

## Customization

You can modify the following parameters in `visual_exponent.py`:
- `angka`: Change the base number (default: 2)
- `range(1, 10001, 20)`: Adjust the exponent range and step size
- Plot labels and styling in the matplotlib section

