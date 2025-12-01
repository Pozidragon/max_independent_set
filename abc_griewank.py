import numpy as np
import matplotlib.pyplot as plt
import time

# Griewank function
def make_griewank(dim: int):
    den = np.sqrt(np.arange(1, dim + 1))

    def f(x: np.ndarray) -> float:
        return 1.0 + np.sum(x ** 2) / 4000.0 - np.prod(np.cos(x / den))

    return f

# ABC Class
class ABC:
    def __init__(
        self,
        func,
        dim=10,
        bounds=(-600.0, 600.0),
        sn=20,  # Number of food sources
        max_iter=1000,
        limit=None,
        hybrid=False,
        c=0.3,  # Hybrid coefficient
        seed=None,
        track_sources=()
    ):
        self.func = func
        self.dim = dim
        self.bounds = np.array(bounds, dtype=float)
        self.span = self.bounds[1] - self.bounds[0]
        self.sn = sn
        self.max_iter = max_iter
        self.hybrid = hybrid
        self.c = c
        self.rng = np.random.default_rng(seed)
        if limit is None:
            self.limit = self.sn * self.dim // 2
        else:
            self.limit = limit

        # Initialize food sources
        self.positions = self.rng.uniform(self.bounds[0], self.bounds[1], (sn, dim))
        self.fitness = np.array([self.func(p) for p in self.positions])
        self.trials = np.zeros(sn, dtype=int)

        self.gbest_idx = np.argmin(self.fitness)
        self.gbest_pos = self.positions[self.gbest_idx].copy()
        self.gbest_val = self.fitness[self.gbest_idx]

        self.history = [self.gbest_val]
        self.traj = {si: [] for si in track_sources}

    def _apply_bounds(self, pos):
        return np.clip(pos, self.bounds[0], self.bounds[1])

    def step(self):
        # Employed bees phase
        for i in range(self.sn):
            k = self.rng.integers(0, self.sn)
            while k == i:
                k = self.rng.integers(0, self.sn)
            j = self.rng.integers(0, self.dim)
            phi = self.rng.uniform(-1, 1)

            new_pos = self.positions[i].copy()
            new_pos[j] = self.positions[i][j] + phi * (self.positions[i][j] - self.positions[k][j])
            if self.hybrid:
                r = self.rng.uniform(0, 1)
                new_pos[j] -= self.c * r * (self.positions[i][j] - self.gbest_pos[j])  # Note the sign

            new_pos = self._apply_bounds(new_pos)
            new_fit = self.func(new_pos)

            if new_fit < self.fitness[i]:
                self.positions[i] = new_pos
                self.fitness[i] = new_fit
                self.trials[i] = 0
            else:
                self.trials[i] += 1

        # Update gbest
        new_gbest_idx = np.argmin(self.fitness)
        if self.fitness[new_gbest_idx] < self.gbest_val:
            self.gbest_val = self.fitness[new_gbest_idx]
            self.gbest_pos = self.positions[new_gbest_idx].copy()

        # Onlooker bees phase
        probs = 1.0 / (1.0 + self.fitness)  # Since minimization, higher fitness for lower f
        probs /= probs.sum()
        for _ in range(self.sn):
            i = np.argmax(self.rng.random() < np.cumsum(probs))
            k = self.rng.integers(0, self.sn)
            while k == i:
                k = self.rng.integers(0, self.sn)
            j = self.rng.integers(0, self.dim)
            phi = self.rng.uniform(-1, 1)

            new_pos = self.positions[i].copy()
            new_pos[j] = self.positions[i][j] + phi * (self.positions[i][j] - self.positions[k][j])
            if self.hybrid:
                r = self.rng.uniform(0, 1)
                new_pos[j] -= self.c * r * (self.positions[i][j] - self.gbest_pos[j])

            new_pos = self._apply_bounds(new_pos)
            new_fit = self.func(new_pos)

            if new_fit < self.fitness[i]:
                self.positions[i] = new_pos
                self.fitness[i] = new_fit
                self.trials[i] = 0
            else:
                self.trials[i] += 1

        # Update gbest again
        new_gbest_idx = np.argmin(self.fitness)
        if self.fitness[new_gbest_idx] < self.gbest_val:
            self.gbest_val = self.fitness[new_gbest_idx]
            self.gbest_pos = self.positions[new_gbest_idx].copy()

        # Scout bees phase
        for i in range(self.sn):
            if self.trials[i] > self.limit:
                self.positions[i] = self.rng.uniform(self.bounds[0], self.bounds[1], self.dim)
                self.fitness[i] = self.func(self.positions[i])
                self.trials[i] = 0

        # Update gbest final
        new_gbest_idx = np.argmin(self.fitness)
        if self.fitness[new_gbest_idx] < self.gbest_val:
            self.gbest_val = self.fitness[new_gbest_idx]
            self.gbest_pos = self.positions[new_gbest_idx].copy()

        self.history.append(self.gbest_val)

        for si in self.traj.keys():
            if 0 <= si < self.sn:
                self.traj[si].append(self.positions[si, :2].copy())

    def run(self):
        t0 = time.time()
        for _ in range(self.max_iter):
            self.step()
        t1 = time.time()
        return {
            "gbest_pos": self.gbest_pos.copy(),
            "gbest_val": float(self.gbest_val),
            "history": np.array(self.history, dtype=float),
            "traj": {k: np.array(v) for k, v in self.traj.items()},
            "time_sec": t1 - t0,
            "evals": self.max_iter * 2 * self.sn  # Approximate, employed + onlookers
        }

def run_multiple_abc(func, hybrid: bool, num_runs: int = 10):
    best_values, times = [], []

    abc_params = {
        "func": func,
        "dim": 10,
        "bounds": (-600, 600),
        "sn": 20,
        "max_iter": 1000,
        "hybrid": hybrid,
        "c": 0.3,
        "track_sources": ()
    }

    label = 'Hybrid ABC' if hybrid else 'Classic ABC'
    print(f"\nRunning {label} ({num_runs} runs)...")

    res_first_run = None

    for i in range(num_runs):
        abc_params["seed"] = 42 + i
        abc_params["track_sources"] = (0, 1, 2) if i == 0 else ()

        abc = ABC(**abc_params)
        res = abc.run()

        if i == 0:
            res_first_run = res

        best_values.append(res["gbest_val"])
        times.append(res["time_sec"])

        print(f"  Run {i + 1}: f* = {res['gbest_val']:.6e}, time = {res['time_sec']:.2f} s")

    best_values_np = np.array(best_values)
    stats = {
        "best_f": np.min(best_values_np),
        "mean_f": np.mean(best_values_np),
        "std_f": np.std(best_values_np),
        "mean_time": np.mean(times),
        "num_evals": abc_params["max_iter"] * 2 * abc_params["sn"]
    }

    return stats, res_first_run

if __name__ == "__main__":
    DIM = 10
    NUM_RUNS = 10
    f = make_griewank(DIM)

    stats_classic, res_classic_first = run_multiple_abc(f, hybrid=False, num_runs=NUM_RUNS)
    stats_hybrid, res_hybrid_first = run_multiple_abc(f, hybrid=True, num_runs=NUM_RUNS)

    print("\n" + "=" * 50)
    print(f"Summary statistics for {NUM_RUNS} runs (f*)")
    print("=" * 50)
    print("Algorithm | Best | Mean | Std | Mean time (s)")
    print("-" * 60)
    print(
        f"Classic ABC | {stats_classic['best_f']:.6e} | {stats_classic['mean_f']:.6e} | "
        f"{stats_classic['std_f']:.6e} | {stats_classic['mean_time']:.3f}"
    )
    print(
        f"Hybrid ABC  | {stats_hybrid['best_f']:.6e} | {stats_hybrid['mean_f']:.6e} | "
        f"{stats_hybrid['std_f']:.6e} | {stats_hybrid['mean_time']:.3f}"
    )
    print("-" * 60)

    # Visualization (first run)
    plt.figure(figsize=(12, 5))

    # Convergence
    plt.subplot(1, 2, 1)
    plt.plot(res_classic_first["history"], label="Classic ABC", linewidth=2)
    plt.plot(res_hybrid_first["history"], label="Hybrid ABC (c=0.3)", linewidth=2)
    plt.yscale('log')
    plt.xlabel("Iteration")
    plt.ylabel("Best value f (log)")
    plt.title("Convergence of ABC on Griewank (run 1)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Trajectories
    plt.subplot(1, 2, 2)
    for si, xy in res_classic_first["traj"].items():
        if xy.shape[0] > 0:
            plt.plot(xy[:, 0], xy[:, 1], linewidth=1.0, alpha=0.8, label=f"Classic:s{si}")
    for si, xy in res_hybrid_first["traj"].items():
        if xy.shape[0] > 0:
            plt.plot(xy[:, 0], xy[:, 1], linewidth=1.0, alpha=0.8, linestyle="--", label=f"Hybrid:s{si}")

    plt.scatter(0, 0, s=60, marker='x', color='red', label="Global min (0,0)")
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.title("Trajectories of selected sources (2D projection, run 1)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)

    plt.tight_layout()
    plt.show()