import pickle
from random import random, sample, shuffle
from math import log

import cv2
import numpy as np


Point = tuple[int, int]


class TravelingSalesman:
    def __init__(self, points: list[Point], color: tuple[int, int, int], t: float = 0.0) -> None:
        self.size = len(points)
        self.points = points
        self.path = list(range(self.size))
        self.length = sum(
            self.dist(self.points[self.path[i]], self.points[self.path[i - 1]])
            for i in range(self.size)
        )
        self.color = color
        self.t = t

    @staticmethod
    def dist(p1: Point, p2: Point) -> float:
        return ((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1])) ** 0.5

    def transit(self) -> bool:
        i, j = sample(range(self.size), 2)
        if i > j:
            i, j = j, i
        diff = 0.0
        diff += self.dist(self.points[j - 1], self.points[i - 1])
        diff += self.dist(self.points[j], self.points[i])
        diff -= self.dist(self.points[i], self.points[i - 1])
        diff -= self.dist(self.points[j], self.points[j - 1])
        if log(random()) * self.t < -diff:
            if i > 0:
                self.points = self.points[:i] + self.points[j - 1 : i - 1 : -1] + self.points[j:]
            elif i == 0:
                self.points = self.points[j - 1 :: -1] + self.points[j:]
            self.length += diff
            return True
        return False

    def draw(self, figure: np.ndarray) -> None:
        tempature_str = f"tempature: {int(self.t)}"
        length_str = f"length: {int(self.length)}"
        npoints_str = f"{self.size} points"
        cv2.putText(
            figure, tempature_str, (0, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1
        )
        cv2.putText(figure, length_str, (0, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(
            figure, npoints_str, (500, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1
        )
        for p1, p2 in zip(self.points, self.points[1:] + [self.points[0]]):
            cv2.line(figure, p1, p2, color=self.color, thickness=2)

    def simulate(self) -> None:
        while True:
            transit = self.transit()
            if not transit:
                continue

            # showing
            figure = np.zeros(shape=(640, 640, 3), dtype=np.uint8)
            self.draw(figure)
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
