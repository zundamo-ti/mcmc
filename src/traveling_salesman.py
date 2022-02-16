import pickle
from random import random, sample, shuffle
from math import exp

import cv2
import numpy as np


Point = tuple[int, int]


class TravelingSalesman:
    def __init__(
        self, points: list[Point], color: tuple[int, int, int], t: float = 0.0
    ) -> None:
        self.size = len(points)
        self.points = points
        self.path = list(range(self.size))
        self.length = sum(
            self.dist(
                self.points[self.path[i]],
                self.points[self.path[i - 1]],
            )
            for i in range(self.size)
        )
        self.color = color
        self.t = t

    @staticmethod
    def dist(p1: Point, p2: Point) -> float:
        return (
            (p1[0] - p2[0]) * (p1[0] - p2[0])
            + (p1[1] - p2[1]) * (p1[1] - p2[1])
        ) ** 0.5

    def transit(self) -> bool:
        i, j = sample(range(self.size), 2)
        if i > j:
            i, j = j, i
        length_diff = self.length_diff(i, j)
        transit_probability = self.transit_probability(length_diff)
        rand = random()
        transit = rand < transit_probability
        if rand < transit_probability:
            if i > 0:
                self.path = (
                    self.path[:i]
                    + self.path[j - 1 : i - 1 : -1]
                    + self.path[j:]
                )
            elif i == 0:
                self.path = self.path[j - 1 :: -1] + self.path[j:]
            self.length += length_diff
        return transit

    def length_diff(self, i: int, j: int) -> float:
        ret = 0.0
        ret += self.dist(
            self.points[self.path[j - 1]],
            self.points[self.path[i - 1]],
        )
        ret += self.dist(
            self.points[self.path[j]],
            self.points[self.path[i]],
        )
        ret -= self.dist(
            self.points[self.path[i]],
            self.points[self.path[i - 1]],
        )
        ret -= self.dist(
            self.points[self.path[j]],
            self.points[self.path[j - 1]],
        )
        return ret

    def transit_probability(self, length_diff: float) -> float:
        try:
            prob = min(exp(-length_diff / self.t), 1)
        except OverflowError:
            prob = 1
        except ZeroDivisionError:
            prob = int(length_diff < 0)
        return prob

    def simulate(self) -> None:
        while True:
            transit = self.transit()
            if not transit:
                continue

            # showing
            loop = self.path.copy()
            loop.append(self.path[0])
            points = [self.points[i] for i in loop]
            figure = np.zeros(
                shape=(640, 640, 3),
                dtype=np.uint8,
            )
            for i in range(self.size + 1):
                cv2.line(
                    figure,
                    points[i - 1],
                    points[i],
                    color=self.color,
                    thickness=2,
                )
            cv2.putText(
                figure,
                f"t: {int(self.t)}",
                (0, 15),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                color=(255, 255, 255),
                thickness=1,
            )
            cv2.putText(
                figure,
                f"length: {int(self.length)}",
                (0, 30),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                color=(255, 255, 255),
                thickness=1,
            )
            cv2.putText(
                figure,
                f"{self.size} points",
                (500, 15),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                color=(255, 255, 255),
                thickness=1,
            )
            cv2.imshow("win", figure)
            key = cv2.waitKey(1)
            if key & 0xFF == 27:
                break
            elif key == ord("k"):
                self.t += 1
            elif key == ord("j"):
                self.t = max(self.t - 1, 0)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-f", "--filename", required=True)
    parser.add_argument("-c", "--color", type=str, help="R,G,B")
    args = parser.parse_args()
    filename: str = f"{args.filename}.pkl"
    color_str: str = args.color
    r, g, b = map(int, color_str.split(","))

    with open(filename, "rb") as f:
        points: list[Point] = pickle.load(f)
        shuffle(points)
        tsp = TravelingSalesman(points, (b, g, r))
        tsp.simulate()
