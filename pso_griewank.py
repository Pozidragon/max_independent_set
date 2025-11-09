import numpy as np
import matplotlib.pyplot as plt
import time

# ========================
#  Функція Griewank (D-вимірна)
# ========================
def make_griewank(dim: int):
    den = np.sqrt(np.arange(1, dim + 1))

    def f(x: np.ndarray) -> float:
        return 1.0 + np.sum(x ** 2) / 4000.0 - np.prod(np.cos(x / den))

    return f


# ========================
#  PSO: класичний та Clerc
# ========================
class PSO:
    def __init__(
            self,
            func,
            dim=10,
            bounds=(-600.0, 600.0),
            swarm_size=40,
            max_iter=1000,
            w=0.729,
            c1=1.494,
            c2=1.494,
            use_constriction=False,
            reflect_on_bounds=False,
            seed=None,
            track_particles=()
    ):
        self.func = func
        self.dim = dim
        self.bounds = np.array(bounds, dtype=float)
        self.span = self.bounds[1] - self.bounds[0]
        self.swarm_size = swarm_size
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.use_constriction = use_constriction
        self.reflect_on_bounds = reflect_on_bounds
        self.rng = np.random.default_rng(seed)

        # Ініціалізація частинок
        self.positions = self.rng.uniform(self.bounds[0], self.bounds[1], (swarm_size, dim))
        self.velocities = self.rng.uniform(-1.0, 1.0, (swarm_size, dim)) * (self.span / 10.0)

        self.pbest_pos = self.positions.copy()
        self.pbest_val = np.array([self.func(p) for p in self.positions])

        gidx = np.argmin(self.pbest_val)
        self.gbest_pos = self.pbest_pos[gidx].copy()
        self.gbest_val = self.pbest_val[gidx]

        self.history = []
        self.traj = {pi: [] for pi in track_particles}

        # Обчислення χ для Clerc
        if self.use_constriction:
            phi = self.c1 + self.c2
            if phi <= 4.0:
                raise ValueError("Для Clerc необхідно, щоб c1 + c2 > 4 (наприклад, c1=c2=2.05).")
            self.chi = 2.0 / abs(2.0 - phi - np.sqrt(phi ** 2 - 4.0 * phi))
        else:
            self.chi = None

    def _apply_bounds(self):
        before = self.positions.copy()
        self.positions = np.clip(self.positions, self.bounds[0], self.bounds[1])

        if self.reflect_on_bounds:
            hit_low = (before < self.bounds[0]) | (self.positions <= self.bounds[0])
            hit_high = (before > self.bounds[1]) | (self.positions >= self.bounds[1])
            self.velocities[hit_low] = np.abs(self.velocities[hit_low])
            self.velocities[hit_high] = -np.abs(self.velocities[hit_high])

    def step(self):
        r1 = self.rng.random((self.swarm_size, self.dim))
        r2 = self.rng.random((self.swarm_size, self.dim))

        if self.use_constriction:
            cognitive = self.chi * self.c1 * r1 * (self.pbest_pos - self.positions)
            social = self.chi * self.c2 * r2 * (self.gbest_pos - self.positions)
            self.velocities = self.chi * self.velocities + cognitive + social
        else:
            cognitive = self.c1 * r1 * (self.pbest_pos - self.positions)
            social = self.c2 * r2 * (self.gbest_pos - self.positions)
            self.velocities = self.w * self.velocities + cognitive + social

        self.positions += self.velocities
        self._apply_bounds()

        values = np.array([self.func(p) for p in self.positions])
        improved = values < self.pbest_val
        if np.any(improved):
            self.pbest_pos[improved] = self.positions[improved]
            self.pbest_val[improved] = values[improved]

        new_gidx = np.argmin(self.pbest_val)
        if self.pbest_val[new_gidx] < self.gbest_val:
            self.gbest_val = self.pbest_val[new_gidx]
            self.gbest_pos = self.pbest_pos[new_gidx].copy()

        self.history.append(self.gbest_val)

        for pi in self.traj.keys():
            if 0 <= pi < self.swarm_size:
                self.traj[pi].append(self.positions[pi, :2].copy())

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
            "evals": self.max_iter * self.swarm_size
        }


def run_multiple_pso(func, is_clerc: bool, num_runs: int = 10):
    """Запускає PSO (класичний або Clerc) кілька разів і збирає статистику."""
    best_values, times = [], []

    pso_params = {
        "func": func,
        "dim": 10,
        "bounds": (-600, 600),
        "swarm_size": 40,
        "max_iter": 1000,
        "use_constriction": is_clerc,
        "reflect_on_bounds": False,
        "track_particles": ()
    }

    if is_clerc:
        pso_params.update({"c1": 2.05, "c2": 2.05})
    else:
        pso_params.update({"w": 0.729, "c1": 1.494, "c2": 1.494})

    print(f"\nЗапуск {'Clerc PSO' if is_clerc else 'Класичного PSO'} ({num_runs} прогонів)...")

    res_first_run = None

    for i in range(num_runs):
        pso_params["seed"] = 42 + i
        pso_params["track_particles"] = (0, 1, 2) if i == 0 else ()

        pso = PSO(**pso_params)
        res = pso.run()

        if i == 0:
            res_first_run = res

        best_values.append(res["gbest_val"])
        times.append(res["time_sec"])

        print(f"  Прогін {i + 1}: f* = {res['gbest_val']:.6e}, час = {res['time_sec']:.2f} с")

    best_values_np = np.array(best_values)
    stats = {
        "best_f": np.min(best_values_np),
        "mean_f": np.mean(best_values_np),
        "std_f": np.std(best_values_np),
        "mean_time": np.mean(times),
        "num_evals": pso_params["max_iter"] * pso_params["swarm_size"]
    }

    return stats, res_first_run


# ========================
#  Основний блок
# ========================
if __name__ == "__main__":
    DIM = 10
    NUM_RUNS = 10
    f = make_griewank(DIM)

    stats_classic, res_classic_first = run_multiple_pso(f, is_clerc=False, num_runs=NUM_RUNS)
    stats_clerc, res_clerc_first = run_multiple_pso(f, is_clerc=True, num_runs=NUM_RUNS)

    print("\n" + "=" * 50)
    print(f"Підсумкова статистика за {NUM_RUNS} прогонів (f*)")
    print("=" * 50)
    print("Алгоритм | Найкраще | Середнє | Std | Середній час (с)")
    print("-" * 60)
    print(
        f"Класика | {stats_classic['best_f']:.6e} | {stats_classic['mean_f']:.6e} | "
        f"{stats_classic['std_f']:.6e} | {stats_classic['mean_time']:.3f}"
    )
    print(
        f"Clerc    | {stats_clerc['best_f']:.6e} | {stats_clerc['mean_f']:.6e} | "
        f"{stats_clerc['std_f']:.6e} | {stats_clerc['mean_time']:.3f}"
    )
    print("-" * 60)

    # Візуалізація результатів (перший прогін)
    plt.figure(figsize=(12, 5))

    # Сходимість
    plt.subplot(1, 2, 1)
    plt.plot(res_classic_first["history"], label="Класичний PSO (ω=0.729, c1=c2=1.494)", linewidth=2)
    plt.plot(res_clerc_first["history"], label="PSO з constriction (Clerc, c1=c2=2.05)", linewidth=2)
    plt.yscale('log')
    plt.xlabel("Ітерація")
    plt.ylabel("Найкраще значення f (log)")
    plt.title("Сходимість PSO на функції Griewank (прогін 1)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Траєкторії
    plt.subplot(1, 2, 2)
    for pi, xy in res_classic_first["traj"].items():
        if xy.shape[0] > 0:
            plt.plot(xy[:, 0], xy[:, 1], linewidth=1.0, alpha=0.8, label=f"Classic:p{pi}")
    for pi, xy in res_clerc_first["traj"].items():
        if xy.shape[0] > 0:
            plt.plot(xy[:, 0], xy[:, 1], linewidth=1.0, alpha=0.8, linestyle="--", label=f"Clerc:p{pi}")

    plt.scatter(0, 0, s=60, marker='x', color='red', label="Глобальний мінімум (0,0)")
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.title("Траєкторії вибраних частинок (2D проєкція, прогін 1)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)

    plt.tight_layout()
    plt.show()

