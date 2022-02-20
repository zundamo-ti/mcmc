import itertools
import random

import cv2
import numpy as np

INF = 1000_000_000_000
COLORS = [
    (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255),
    )
    for _ in range(100)
]
Job = int
Machine = int
Vertex = tuple[Job, Machine]


class SimpleShopScheduling:
    def __init__(
        self,
        n_jobs: int,
        n_machines: int,
        cost: list[list[int]],
        tempature: int,
    ) -> None:
        """
        Args
        - `n_jobs: int`
        - `n_machines: int`
        - `cost: list[list[int]]`
          - `cost[job][machine]`がコストを表す
        - `tempature: int`
        """
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.vertices: list[Vertex] = list(
            itertools.product(range(n_jobs), range(n_machines))
        )
        self.cost: dict[Vertex, int] = {
            (job, machine): cost[job][machine]
            for job, machine in self.vertices
        }
        self.job_orders: list[list[Job]] = [
            list(range(n_jobs)) for _ in range(n_machines)
        ]
        self.machine_orders: list[list[Machine]] = [
            list(range(n_machines)) for _ in range(n_jobs)
        ]
        makespan, start_times = self.__calc()
        self.makespan = makespan
        self.start_times = start_times
        self.best_possible = max(
            max(
                sum(cost[job][machine] for job in range(self.n_jobs))
                for machine in range(self.n_machines)
            ),
            max(
                sum(cost[job][machine] for machine in range(self.n_machines))
                for job in range(self.n_jobs)
            ),
        )
        self.tempature = tempature

    def __calc(self) -> tuple[int, dict[Vertex, int]]:
        START: Vertex = (-1, 0)
        FINISH: Vertex = (0, -1)
        G: dict[Vertex, list[tuple[Vertex, int]]] = {
            v: [] for v in self.vertices
        }
        G[START] = []
        G[FINISH] = []
        for j, m in self.vertices:
            G[START].append(((j, m), 0))
            G[j, m].append((FINISH, -self.cost[j, m]))
            if j < self.n_jobs - 1:
                j1, j2 = self.job_orders[m][j : j + 2]
                G[j1, m].append(((j2, m), -self.cost[j1, m]))
            if m < self.n_machines - 1:
                m1, m2 = self.machine_orders[j][m : m + 2]
                G[j, m1].append(((j, m2), -self.cost[j, m1]))
        dist: dict[Vertex, int] = {v: INF for v in self.vertices}
        dist[START] = 0
        dist[FINISH] = INF
        update = True
        for _ in range(len(self.vertices) + 2):
            update = False
            for s, edges in G.items():
                for t, c in edges:
                    if dist[s] + c < dist[t]:
                        dist[t] = dist[s] + c
                        update = True
            if not update:
                break
        for s, edges in G.items():
            for t, c in edges:
                if dist[s] + c < dist[t]:
                    return INF, {}
        makespan = -dist[FINISH]
        start_times = {v: -dist[v] for v in self.vertices}
        return makespan, start_times

    def transit(self) -> bool:
        if random.random() < 0.5:
            m = random.choice(range(self.n_machines))
            j1, j2 = random.sample(range(self.n_jobs), 2)
            order = self.job_orders[m]
            order[j1], order[j2] = order[j2], order[j1]
            makespan, start_times = self.__calc()
            diff = makespan - self.makespan
            if np.log(np.random.random()) * self.tempature <= -diff:
                self.makespan = makespan
                self.start_times = start_times
                return True
            order[j1], order[j2] = order[j2], order[j1]
            return False
        else:
            j = random.choice(range(self.n_jobs))
            m1, m2 = random.sample(range(self.n_machines), 2)
            order = self.machine_orders[j]
            order[m1], order[m2] = order[m2], order[m1]
            makespan, start_times = self.__calc()
            diff = makespan - self.makespan
            if np.log(np.random.random()) * self.tempature <= -diff:
                self.makespan = makespan
                self.start_times = start_times
                return True
            order[m1], order[m2] = order[m2], order[m1]
            return False

    def draw(self, image: np.ndarray) -> None:
        padding_top = 50
        width = 1
        height = 20
        for job, machine in itertools.product(
            range(self.n_jobs), range(self.n_machines)
        ):
            top = int(machine * height + padding_top)
            bottom = top + int(height)
            left = int(self.start_times[job, machine] * width)
            right = left + int(self.cost[job, machine] * width)
            cv2.rectangle(
                image,
                (left, top),
                (right, bottom),
                COLORS[job % len(COLORS)],
                -1,
            )
        cv2.putText(
            image,
            f"makespan: {int(self.makespan)}",
            (0, 15),
            cv2.FONT_HERSHEY_COMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            image,
            f"tempature: {self.tempature}",
            (0, 35),
            cv2.FONT_HERSHEY_COMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            image,
            f"best possible: {self.best_possible}",
            (150, 15),
            cv2.FONT_HERSHEY_COMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    def simulate(self, shape: tuple[int, int]) -> None:
        initial_value = self.makespan
        print("initial value:\t", initial_value)
        best_value = initial_value
        count = 0
        while True:
            self.transit()
            best_value = min(best_value, self.makespan)
            count += 1
            if count % 100 != 0:
                continue
            image = np.zeros(shape=shape + (3,), dtype=np.uint8)
            self.draw(image)
            cv2.imshow("simple shop scheduling", image)
            key = cv2.waitKey(1)
            if key & 0xFF == 27:
                break
            if key == ord("j"):
                self.tempature = max(0, self.tempature - 1)
            if key == ord("k"):
                self.tempature += 1
        initial_difference = initial_value - self.best_possible
        best_difference = best_value - self.best_possible
        print(
            "best value"
            f"\t{best_value}\n"
            "difference with best possible\n"
            f"\t{initial_difference} -> {best_difference}\n"
            "imporved\n"
            f"\t{int(100 * (1 - best_difference / initial_difference))}%\n"
            "differece rate with best possible\n"
            f"\t{int(100 * best_difference / self.best_possible)}%"
        )
        cv2.destroyAllWindows()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-j", "--n-jobs", type=int, default=5)
    parser.add_argument("-m", "--n-machines", type=int, default=5)
    parser.add_argument("-t", "--tempature", type=int, default=0)
    args = parser.parse_args()
    n_jobs: int = args.n_jobs
    n_machines: int = args.n_machines
    tempature: int = args.tempature
    cost = [
        [random.randint(1, 50) for _ in range(n_machines)]
        for _ in range(n_jobs)
    ]
    sss = SimpleShopScheduling(n_jobs, n_machines, cost, tempature)
    sss.simulate(shape=(50 + 20 * n_machines, 35 * n_jobs + 35 * n_machines))
