import itertools
from math import log
from random import random

import cv2
import numpy as np


COLOR = ((255, 64, 0), (0, 192, 255))


class IsingModel2D:
    def __init__(self, size: int, j: float, k: float) -> None:
        self.size = size
        self.j = j
        self.k = k
        self.spins = 1 - 2 * np.random.randint(2, size=(size, size))
        self.hamiltonian = 0
        self.tempature = 0.0

    def transit(self) -> bool:
        i, j = np.random.randint(self.size, size=2) - 1
        diff = 0
        diff += self.spins[i - 1, j]
        diff += self.spins[i + 1, j]
        diff += self.spins[i, j - 1]
        diff += self.spins[i, j + 1]
        diff *= 2 * self.j * self.spins[i, j]
        diff += 2 * self.k * self.spins[i, j]
        if log(random()) * self.tempature <= -diff:
            self.spins[i, j] *= -1
            self.hamiltonian += diff
            return True
        return False

    def draw(self, image: np.ndarray, scale: int, padding: int) -> None:
        for i, j in itertools.product(range(self.size), repeat=2):
            x = i * scale + padding
            y = j * scale + padding
            cv2.circle(
                image,
                center=(x, y),
                radius=scale // 2,
                color=COLOR[(self.spins[i, j] + 1) // 2],
                thickness=-1,
            )

    def simulate(self) -> None:
        count = 0
        speed = self.size**2
        while True:
            self.transit()
            count += 1
            if count % speed != 0:
                continue
            scale = 20
            padding = 20
            image = np.full(
                shape=(
                    scale * self.size + 4 * padding,
                    scale * self.size + 2 * padding,
                    3,
                ),
                fill_value=255,
                dtype="uint8",
            )
            self.draw(image, scale, padding)
            cv2.putText(
                image,
                f"tempature: {int(self.tempature * 100 + 0.5) / 100}",
                (0, scale * self.size + 2 * padding),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=2,
                color=(0, 0, 0),
                thickness=2,
            )
            cv2.imshow("ising", image)
            key = cv2.waitKey(1)
            if key & 0xFF == 27:
                break
            if key & 0xFF == ord("j"):
                self.tempature = max(self.tempature - 0.1, 0)
            if key & 0xFF == ord("k"):
                self.tempature += 0.1


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-s", "--s", type=int, default=100)
    parser.add_argument("-j", "--j", type=float, default=1)
    parser.add_argument("-k", "--k", type=float, default=0)
    args = parser.parse_args()
    model = IsingModel2D(args.s, args.j, args.k)
    model.simulate()
