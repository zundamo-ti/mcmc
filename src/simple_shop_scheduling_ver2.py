import itertools
import random

import cv2
import numpy as np
import pulp

COLORS = [
    (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255),
    )
    for _ in range(100)
]


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
        self.vertices = list(
            itertools.product(range(n_jobs), range(n_machines))
        )
        self.cost = {
            (job, machine): cost[job][machine]
            for job, machine in self.vertices
        }
        self.order = list(range(n_machines))
        makespan, start_times = self.__calc()
        self.makespan = makespan
        self.start_times = start_times
        self.tempature = tempature

    def __calc_fast(self) -> tuple[int, dict[tuple[int, int], int]]:
        raise NotImplementedError

    def __calc(self) -> tuple[int, dict[tuple[int, int], int]]:
        problem: pulp.LpProblem = pulp.LpProblem(
            "simpla shop scheduling", pulp.LpMinimize
        )
        start_variables = {
            (job, machine): pulp.LpVariable(
                f"start of {job},{machine}", cat=pulp.LpInteger
            )
            for job, machine in itertools.product(
                range(n_jobs), range(n_machines)
            )
        }
        finish = pulp.LpVariable("finish", cat=pulp.LpInteger)
        problem += finish
        for machine in range(self.n_machines):
            problem += 0 <= start_variables[0, machine]
            problem += (
                start_variables[self.n_jobs - 1, machine]
                + self.cost[self.n_jobs - 1, machine]
                <= finish
            )
        for job, machine in itertools.product(
            range(self.n_jobs - 1), range(self.n_machines)
        ):
            problem += (
                start_variables[job, machine] + self.cost[job, machine]
                <= start_variables[job + 1, machine]
            )
        for i in range(self.n_machines - 1):
            machine1, machine2 = self.order[i : i + 2]
            for job in range(self.n_jobs):
                problem += (
                    start_variables[job, machine1] + self.cost[job, machine1]
                    <= start_variables[job, machine2]
                )
        solver = pulp.PULP_CBC_CMD(timeLimit=1, gapRel=0.01, msg=False)
        status = problem.solve(solver)
        assert status == pulp.LpStatusOptimal
        makespan: int = finish.varValue
        start_times: dict[tuple[int, int], int] = {
            (job, machine): start_variables[job, machine].varValue
            for job, machine in start_variables
        }
        return makespan, start_times

    def transit(self) -> bool:
        i, j = random.sample(range(self.n_machines), 2)
        self.order[i], self.order[j] = self.order[j], self.order[i]
        makespan, start_times = self.__calc()
        diff = makespan - self.makespan
        if np.log(np.random.random()) * self.tempature <= -diff:
            self.makespan = makespan
            self.start_times = start_times
            return True
        self.order[i], self.order[j] = self.order[j], self.order[i]
        return False

    def draw(self, image: np.ndarray) -> None:
        padding_top = 50
        padding_left = 50
        width = 1
        height = 20
        for job, (index, machine) in itertools.product(
            range(self.n_jobs), enumerate(self.order)
        ):
            top = int(index * height + padding_top)
            bottom = top + int(height)
            left = int(self.start_times[job, machine] * width + padding_left)
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
                f"{machine}",
                (0, bottom),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
        cv2.putText(
            image,
            f"makespan: {self.makespan}",
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

    def simulate(self, shape: tuple[int, int]) -> None:
        initial_value = self.makespan
        print("initial value:\t", initial_value)
        best_value = initial_value
        while True:
            self.transit()
            best_value = min(best_value, self.makespan)
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
        print("best value:\t", best_value)
        print(
            "improved ratio:\t",
            int(100 * (1 - best_value / initial_value)),
            "%",
        )
        cv2.destroyAllWindows()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-j", "--n-jobs", type=int, default=10)
    parser.add_argument("-m", "--n-machines", type=int, default=20)
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
