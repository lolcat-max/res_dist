import numpy as np
import math
import warnings
import sys
import random

sys.setrecursionlimit(2000000)
warnings.filterwarnings("ignore")

# ==========================================
# 1. ASTRONOMICAL PHYSICS KERNEL
# ==========================================

class AstroDomain:
    def __init__(self, name, initial_scale=10.0):
        self.name = name
        self.val = initial_scale
        self.velocity = 0.0
        
    def update_multiplicative(self, factor, dt):
        """Updates value by a multiplicative factor (safe for huge numbers)."""
        target_velocity = factor
        self.velocity = (self.velocity * 0.8) + (target_velocity * 0.2)
        step_change = np.clip(self.velocity * dt, -0.1, 0.1)
        try:
            self.val *= (1.0 + step_change)
        except OverflowError:
            self.val = float('inf')
        if self.val < 1e-100:
            self.val = 1e-100

# ==========================================
# 2. LOG-SCALE MATH ENGINE
# ==========================================

class AstroPhysicsSolver:
    def __init__(self):
        self.variables = {}
        
    def create_var(self, name, rough_magnitude):
        self.variables[name] = AstroDomain(name, initial_scale=rough_magnitude)
    
    def _find_integer_factors(self, target_int, approx_x, approx_y, search_radius=10_000):
        if target_int <= 0:
            return None
        if approx_x > approx_y:
            approx_x, approx_y = approx_y, approx_x
        best_pair = None
        min_diff = float('inf')
        sqrt_n = math.isqrt(target_int)
        start = max(1, int(approx_x) - search_radius)
        end = min(int(approx_x) + search_radius, sqrt_n)
        for cand_x in range(end, start - 1, -1):
            if target_int % cand_x == 0:
                cand_y = target_int // cand_x
                if cand_x <= cand_y:
                    diff = abs(cand_x - approx_x) + abs(cand_y - approx_y)
                    if diff < min_diff:
                        min_diff = diff
                        best_pair = (cand_x, cand_y)
        if best_pair is None:
            for i in range(sqrt_n, 0, -1):
                if target_int % i == 0:
                    j = target_int // i
                    if i <= j:
                        best_pair = (i, j)
                    break
            if best_pair is None:
                best_pair = (1, target_int)
        return best_pair
    
    def _solve_subset_sum_exact(self, numbers, target):
        if target == 0:
            return []
        n = len(numbers)
        if n == 0 or target < 0:
            return None
        if target > 100000:
            return None
        dp = [False] * (target + 1)
        dp[0] = True
        prev = [-1] * (target + 1)
        for num in numbers:
            if num > target:
                continue
            for s in range(target, num - 1, -1):
                if not dp[s] and dp[s - num]:
                    dp[s] = True
                    prev[s] = s - num
        if not dp[target]:
            return None
        subset = []
        s = target
        while s > 0:
            prev_s = prev[s]
            if prev_s == -1:
                break
            num = s - prev_s
            subset.append(num)
            s = prev_s
        return subset
    
    def _solve_subset_sum_annealing(self, numbers, target, steps=50_000):
        n = len(numbers)
        self.variables = {}
        
        for i in range(n):
            var_name = f'incl_{i}'
            self.create_var(var_name, rough_magnitude=0.5)
        
        for _ in range(steps):
            vals = {n: d.val for n, d in self.variables.items()}
            current_sum = sum(vals[f'incl_{i}'] * numbers[i] for i in range(n))
            error = abs(current_sum - target)
            if error < 1e-6:
                break
            perturbation = 0.01
            for i in range(n):
                name = f'incl_{i}'
                domain = self.variables[name]
                orig = domain.val
                clamped_orig = np.clip(orig, 0.0, 1.0)
                domain.val = min(1.0, clamped_orig + perturbation)
                sum_new_up = sum(self.variables[f'incl_{j}'].val * numbers[j] for j in range(n))
                sens_up = (sum_new_up - current_sum) / perturbation if perturbation > 0 else 0
                domain.val = max(0.0, clamped_orig - perturbation)
                sum_new_down = sum(self.variables[f'incl_{j}'].val * numbers[j] for j in range(n))
                sens_down = (sum_new_down - current_sum) / (-perturbation) if perturbation > 0 else 0
                sensitivity = (sens_up + sens_down) / 2.0
                if abs(sensitivity) < 1e-6:
                    sensitivity = numbers[i]
                force = (target - current_sum) / sensitivity if sensitivity != 0 else 0
                force *= 0.1
                domain.update_multiplicative(force, dt=0.01)
                domain.val = np.clip(domain.val, 0.0, 1.0)
        
        inclusions = {
            name: round(np.clip(val, 0.0, 1.0))
            for name, val in {n: d.val for n, d in self.variables.items()}.items()
            if name.startswith('incl_')
        }
        subset = [numbers[i] for i in range(n) if inclusions[f'incl_{i}'] == 1]
        approx_sum = sum(subset)
        return subset if approx_sum == target else None
    
    def _solve_subset_sum_annealing_fast(self, numbers, target, steps=100):
        """Ultra-fast annealing for Sudoku heuristics."""
        n = len(numbers)
        if n == 0:
            return None
        
        self.variables = {}
        
        initial_prob = min(0.8, target / sum(numbers)) if sum(numbers) > 0 else 0.5
        for i in range(n):
            var_name = f'incl_{i}'
            self.create_var(var_name, rough_magnitude=initial_prob)
        
        learning_rate = 0.5
        
        for step in range(steps):
            vals = {n: d.val for n, d in self.variables.items()}
            current_sum = sum(vals[f'incl_{i}'] * numbers[i] for i in range(n))
            error = current_sum - target
            
            if abs(error) < 0.1:
                break
            
            adaptive_lr = learning_rate / (1.0 + step / 20.0)
            
            for i in range(n):
                name = f'incl_{i}'
                domain = self.variables[name]
                gradient = error * numbers[i]
                domain.val = np.clip(domain.val - adaptive_lr * gradient / max(abs(target), 1.0), 0.0, 1.0)
        
        vals_final = {n: d.val for n, d in self.variables.items()}
        inclusions = [(i, vals_final[f'incl_{i}']) for i in range(n)]
        inclusions.sort(key=lambda x: -x[1])
        
        return [i for i, _ in inclusions[:len(inclusions)//2]]
    
    def solve(self, equation, steps=1_000_000, prefer_integers=False,
              subset_numbers=None, subset_target=None):
        if subset_numbers is not None and subset_target is not None:
            exact_subset = self._solve_subset_sum_exact(subset_numbers, subset_target)
            if exact_subset:
                return {'subset': sorted(exact_subset), 'method': 'exact_dp'}
            print("[Exact DP] No solution found.")
            anneal_subset = self._solve_subset_sum_annealing(subset_numbers, subset_target, steps)
            if anneal_subset:
                print(f"[Annealing Solution] Subset: {sorted(anneal_subset)} "
                      f"(sum: {sum(anneal_subset)})")
                return {'subset': sorted(anneal_subset), 'method': 'annealing'}
            print("[Subset Sum] No solution found.")
            return {'subset': None, 'method': 'failed'}
        
        if not equation:
            return {}
        
        print(f"\n[Physics Engine] Target Equation: {equation}")
        lhs_str, rhs_str = equation.split('=')
        
        target_int = None
        target_val = None
        rhs_stripped = rhs_str.strip()
        try:
            if 'e' in rhs_stripped.lower() or '.' in rhs_stripped:
                target_val = float(eval(rhs_stripped))
                if target_val.is_integer():
                    target_int = int(target_val)
            else:
                target_int = int(rhs_stripped)
                target_val = float(target_int)
        except (ValueError, OverflowError):
            try:
                target_val = float(eval(rhs_stripped))
                if target_val.is_integer():
                    target_int = int(target_val)
            except Exception:
                target_val = float('inf')
                target_int = None
            print("Warning: Target parsing issues, using approximate.")
        
        if target_val is None or target_val == float('inf'):
            print("[System] Target too large or invalid. Stopping.")
            return {}
        if target_val > 0:
            log_target = math.log10(target_val)
        else:
            log_target = -100
        print(f"[System] Target Magnitude: 10^{log_target:.2f}")
        if target_int is not None:
            print(f"[System] Target is exact integer: {target_int}")
        else:
            print(f"[System] Target approximate: {target_val}")
        
        import re
        tokens = set(re.findall(r'[a-zA-Z_]+', lhs_str))
        num_vars = len(tokens) if len(tokens) > 0 else 1
        estimated_scale = 10 ** (log_target / num_vars)
        for t in tokens:
            if t not in self.variables:
                self.create_var(t, rough_magnitude=estimated_scale)
        
        for _ in range(steps):
            vals = {n: d.val for n, d in self.variables.items()}
            try:
                current_lhs = eval(lhs_str, {}, vals)
            except OverflowError:
                current_lhs = float('inf')
            if current_lhs <= 0:
                current_lhs = 1e-100
            try:
                log_current = math.log10(current_lhs)
            except ValueError:
                log_current = -100
            error = log_current - log_target
            if abs(error) < 1e-8:
                break
            perturbation = 1.001
            log_perturb_delta = math.log10(perturbation)
            for name in tokens:
                domain = self.variables[name]
                orig = domain.val
                domain.val = orig * perturbation
                vals_new = {n: v.val for n, v in self.variables.items()}
                try:
                    lhs_new = eval(lhs_str, {}, vals_new)
                    if lhs_new <= 0:
                        lhs_new = 1e-100
                    log_new = math.log10(lhs_new)
                except Exception:
                    log_new = log_current
                sensitivity = (log_new - log_current) / log_perturb_delta
                domain.val = orig
                if abs(sensitivity) < 0.001:
                    sensitivity = 1.0
                force = -error / sensitivity
                force *= 10.0
                domain.update_multiplicative(force, dt=0.01)
        
        float_res = {n: d.val for n, d in self.variables.items()}
        if prefer_integers and target_int is not None and len(tokens) == 2:
            if 'x' in tokens and 'y' in tokens:
                approx_x = float_res['x']
                approx_y = float_res['y']
                int_pair = self._find_integer_factors(target_int, approx_x, approx_y)
                if int_pair:
                    a, b = int_pair
                    if abs(a - approx_x) < abs(b - approx_x):
                        float_res['x'] = a
                        float_res['y'] = b
                    else:
                        float_res['x'] = b
                        float_res['y'] = a
                    print(f"[Integer Mode] Found factors: {a} * {b} = {target_int}")
        return float_res

# ==========================================
# 3. FAST SUDOKU SOLVER
# ==========================================

N = 64
BOX = int(math.isqrt(N))

class GeneralSudokuSolver:
    def __init__(self, engine):
        self.engine = engine
        self.steps = 0
        self.use_astro_every = 20

    def _ok(self, grid, r, c, d):
        for k in range(N):
            if grid[r][k] == d or grid[k][c] == d:
                return False
        br, bc = BOX * (r // BOX), BOX * (c // BOX)
        for rr in range(br, br + BOX):
            for cc in range(bc, bc + BOX):
                if grid[rr][cc] == d:
                    return False
        return True

    def _find_empty_mrv(self, grid):
        """Find empty cell with minimum remaining values."""
        best = None
        min_count = N + 1
        for r in range(N):
            for c in range(N):
                if grid[r][c] == 0:
                    count = sum(1 for d in range(1, N + 1) if self._ok(grid, r, c, d))
                    if count == 0:
                        return r, c, []
                    if count < min_count:
                        min_count = count
                        best = (r, c)
        if best:
            r, c = best
            cands = [d for d in range(1, N + 1) if self._ok(grid, r, c, d)]
            return r, c, cands
        return None, None, []

    def _choose_order(self, grid, r, c, candidates):
        """Order candidates using fast heuristic or occasional astro."""
        if len(candidates) <= 1:
            return candidates
        
        self.steps += 1
        
        if self.steps % 1000 == 0:
            print(f"  Steps: {self.steps}")
        
        # Simple degree heuristic most of the time
        if self.steps % self.use_astro_every != 0:
            costs = []
            for d in candidates:
                cost = 0
                for k in range(N):
                    if grid[r][k] == 0:
                        cost += 1
                    if grid[k][c] == 0:
                        cost += 1
                costs.append((d, cost))
            costs.sort(key=lambda x: -x[1])
            return [d for d, _ in costs]
        
        # Occasionally use astro
        costs = []
        for d in candidates:
            cost = 0
            for k in range(N):
                if grid[r][k] == 0 and k != c and self._ok(grid, r, k, d):
                    cost += 1
                if grid[k][c] == 0 and k != r and self._ok(grid, k, c, d):
                    cost += 1
            br, bc = BOX * (r // BOX), BOX * (c // BOX)
            for rr in range(br, br + BOX):
                for cc in range(bc, bc + BOX):
                    if grid[rr][cc] == 0 and (rr != r or cc != c) and self._ok(grid, rr, cc, d):
                        cost += 1
            costs.append(max(1, cost))
        
        total = sum(costs)
        target = max(1, total // 2)
        
        selected_indices = self.engine._solve_subset_sum_annealing_fast(costs, target, steps=50)
        
        if selected_indices:
            remaining = [i for i in range(len(candidates)) if i not in selected_indices]
            return [candidates[i] for i in selected_indices] + [candidates[i] for i in remaining]
        
        paired = list(zip(candidates, costs))
        paired.sort(key=lambda x: -x[1])
        return [d for d, _ in paired]

    def _backtrack(self, grid):
        r, c, cand = self._find_empty_mrv(grid)
        if r is None:
            return True
        if not cand:
            return False
        
        ordered = self._choose_order(grid, r, c, cand)
        for d in ordered:
            grid[r][c] = d
            if self._backtrack(grid):
                return True
            grid[r][c] = 0
        return False

    def solve(self, grid):
        g = [row[:] for row in grid]
        self.steps = 0
        
        empty = sum(1 for r in range(N) for c in range(N) if g[r][c] == 0)
        print(f"\nSolving {N}x{N} Sudoku ({empty} empty cells)...")
        
        if self._backtrack(g):
            print(f"SOLVED in {self.steps} steps")
            return g
        print(f"FAILED after {self.steps} steps")
        return None

def get_standard_puzzle(n):
    if n == 9:
        return [
            [5,3,0, 0,7,0, 0,0,0],
            [6,0,0, 1,9,5, 0,0,0],
            [0,9,8, 0,0,0, 0,6,0],
            [8,0,0, 0,6,0, 0,0,3],
            [4,0,0, 8,0,3, 0,0,1],
            [7,0,0, 0,2,0, 0,0,6],
            [0,6,0, 0,0,0, 2,8,0],
            [0,0,0, 4,1,9, 0,0,5],
            [0,0,0, 0,8,0, 0,7,9],
        ]
    elif n == 16:
        return [
            [1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
            [0,0,0,0, 2,0,0,0, 0,0,0,0, 3,0,0,0],
            [0,0,0,0, 0,0,0,0, 0,4,0,0, 0,0,0,0],
            [0,0,0,0, 0,5,0,0, 0,0,0,0, 0,0,0,6],
            [0,7,0,0, 0,0,0,0, 0,0,8,0, 0,0,0,0],
            [0,0,0,0, 0,0,0,9, 0,0,0,0, 0,0,0,0],
            [0,0,0,0,10,0,0,0, 0,0,0,0, 0,0,0,0],
            [0,0,0,0, 0,0,0,0,11,0,0,0, 0,0,0,0],
            [0,0,0,0, 0,0,12,0, 0,0,0,0, 0,0,0,0],
            [0,0,0,0, 0,0,0,0, 0,0,0,13,0,0,0,0],
            [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,14,0,0],
            [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
            [0,0,0,0, 0,0,0,0, 0,0,0,0,15,0,0,0],
            [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,16,0],
            [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
            [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
        ]
    else:
        return [[0]*n for _ in range(n)]

# ==========================================
# 4. MAIN
# ==========================================
if __name__ == "__main__":
    N = 16  
    BOX = int(math.isqrt(N))
        
    engine = AstroPhysicsSolver()
    puzzle = get_standard_puzzle(N)
    puzzle[9][9] = 9
    
    sudoku_solver = GeneralSudokuSolver(engine)
    solution = sudoku_solver.solve(puzzle)
    
    print(f"\nSudoku solution ({N}x{N}):")
    if solution:
        print("First 10x10 region:")
        for row in solution:
            print(" ".join(f"{v:2}" for v in row))
    else:
        print("No solution found.")
