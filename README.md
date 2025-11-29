AstroPhysics Sudoku Kernel
==========================

A hybrid astronomical-physics inspired constraint engine with a general N×N Sudoku solver (supporting 9×9, 16×16, 25×25, …) and a subset-sum annealing demo, all in one script.[3]

## Features

- AstroDomain “physics” variables with multiplicative, velocity-damped updates to keep huge-number computations numerically stable.[5]
- AstroPhysicsSolver:
  - Log-scale error minimization for arbitrary equations.[3]
  - Exact subset-sum solver via dynamic programming.[3]
  - Heuristic subset-sum “annealing” using bounded continuous variables in \([0,1]\).[5]
  - Optional integer-factor refinement for simple \(x \cdot y = N\) style equations.[3]
- GeneralSudokuSolver:
  - Works for any \(N\) that is a perfect square (e.g., 9, 16, 25, 64).  
  - Standard backtracking with row/column/box checks.[3]
  - Hook to use the AstroPhysicsSolver for candidate ordering (currently a stub but easy to extend).  
- Demo main:
  - Builds and prints an arbitrary 8×8 array.  
  - Instantiates a 64×64 “Sudoku” solver while still using a 16×16-style starter puzzle, demonstrating scalability vs. realistic puzzle sizing.  

## Project structure

This repository currently consists of a single Python script:

- `astro_sudoku.py`  
  - `AstroDomain`: physics-like variable domain with velocity and multiplicative updates.  
  - `AstroPhysicsSolver`: log-scale solver, subset-sum (exact + annealing), and integer factor helper.  
  - `GeneralSudokuSolver`: generic N×N Sudoku backtracking solver.  
  - `get_standard_puzzle(N)`: returns a small demo puzzle for \(N=9\) or \(N=16\); otherwise an empty grid.  
  - `__main__` demo: array print + Sudoku solve attempt.  

## Requirements

- Python 3.9+ (tested in standard CPython).[3]
- NumPy for numeric stability and clipping:
  - `numpy` (>=1.20 recommended).[8]

Install dependencies:

```bash
pip install numpy
```

## Usage

Run the demo script:

```bash
python astro_sudoku.py
```

This will:

- Print an 8×8 “generic array” with a few manually edited entries.  
- Construct an `AstroPhysicsSolver` instance and a `GeneralSudokuSolver`.  
- Attempt to solve a Sudoku with current `N` and `BOX` values in `__main__`.  

To switch Sudoku size:

1. Change the top-level `N` and derived `BOX`:

```python
N = 16
BOX = int(math.isqrt(N))
```

2. Adjust the `__main__` section accordingly:

```python
if __name__ == "__main__":
    engine = AstroPhysicsSolver()
    N = 16
    BOX = int(math.isqrt(N))
    sudoku_solver = GeneralSudokuSolver(engine)
    puzzle = get_standard_puzzle(N)
    solution = sudoku_solver.solve(puzzle)
    ...
```

For a custom puzzle:

```python
custom_puzzle = [[0]*N for _ in range(N)]
# Fill in givens as integers 1..N, 0 for empty
solution = sudoku_solver.solve(custom_puzzle)
```

## Subset-sum and physics engine

You can use `AstroPhysicsSolver` independently of Sudoku:

```python
engine = AstroPhysicsSolver()

# Exact + annealing subset-sum
numbers = [3, 7, 10, 25, 50]
target = 40
result = engine.solve("", subset_numbers=numbers, subset_target=target)
print(result)
```

For equation solving (continuous):

```python
engine = AstroPhysicsSolver()
res = engine.solve("x * y", prefer_integers=True)  # With a numeric RHS in the string, e.g. "x*y=1234567"
print(res)
```

Note: For equation solving, the `equation` argument must include both LHS and RHS, e.g. `"x*y=1234567"`. The solver uses log-scale error minimization, then optionally refines integer factors for two-variable cases.[3]

## Extending candidate ordering

`GeneralSudokuSolver._choose_order_with_astro` currently calls the physics engine as a placeholder. To make it meaningful:

- Replace the simple `costs = [1] * len(candidates)` with heuristic costs (e.g., remaining legal moves or entropy of each candidate), then let the subset/physics engine rank or select candidates.[5]
- You can also inject randomness or temperature-like behaviour from the annealing routine for stochastic search.  

## License

This project is provided as-is, with full respect for software licensing and copyright.  
If you integrate third-party libraries or puzzles, ensure their licenses are compatible and properly attributed.

[1](https://stackoverflow.com/questions/46312470/difference-between-methods-and-attributes-in-python)
[2](https://www.turing.com/kb/introduction-to-python-class-attributes)
[3](https://docs.python.org/3/tutorial/classes.html)
[4](https://www.geeksforgeeks.org/python/accessing-attributes-methods-python/)
[5](https://realpython.com/python-classes/)
[6](https://github.com/Reviewable/demo/blob/master/astropy/cosmology/funcs.py)
[7](https://www.almabetter.com/bytes/tutorials/python/methods-and-attributes-in-python)
[8](https://numpy.org/doc/stable/user/troubleshooting-importerror.html)
[9](https://www.youtube.com/watch?v=tQ1n-ySubAM)
[10](https://spacepy.github.io/_modules/spacepy/toolbox.html)
[11](https://www.w3schools.com/python/python_classes.asp)
[12](https://github.com/sczesla/PyAstronomy/blob/master/setup.py)
[13](https://asd.gsfc.nasa.gov/xassist/pipeline4/chandra/12249/acisf12249/analysis/spatial/fullrun_ccd0_acisf12249_pi14-548_cl_pass0_tfrozen_src.sh.log)
[14](https://issm.ess.uci.edu/trac/issm/changeset/26635/issm/trunk-jpl)
[15](https://numpy.org/devdocs/reference/random/extending.html)
[16](https://numpy.org/doc/stable/user/misc.html)
[17](https://numpy.org/devdocs/reference/random/index.html)
