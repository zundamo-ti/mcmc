import itertools
from math import log

import cv2
import numpy as np


COLOR = ((255, 64, 0), (0, 192, 255))


class IsingModel2D:
    def __init__(self, size: int, j: float, k: float) -> None:
        self.size = size
        self.j = j
        self.k = k
        self.spins = 1 - 2 * np.random.randint(2, size=(size, size))
        self.tempature = 0.0

    @property
    def magnetic(self) -> float:
        return float(self.spins.mean())

    def transit(self) -> bool:
        i, j = np.random.randint(self.size, size=2) - 1
        diff = 0
        diff += self.spins[i - 1, j]
        diff += self.spins[i + 1, j]
        diff += self.spins[i + 1, j + 1]
        diff += self.spins[i, j - 1]
        diff += self.spins[i, j + 1]
        diff += self.spins[i - 1, j - 1]
        diff *= 2 * self.j * self.spins[i, j]
        diff += 2 * self.k * self.spins[i, j]
        if log(np.random.rand()) * self.tempature <= -diff:
            self.spins[i, j] *= -1
            return True
        return False

    def draw(self, image: np.ndarray, padding: int) -> None:
        for i, j in itertools.product(range(self.size), repeat=2):
            x = (i * 22 - j * 11 + padding) % (self.size * 22)
            y = j * 19 + padding
            cv2.circle(
                image,
                center=(x, y),
                radius=11,
                color=COLOR[(self.spins[i, j] + 1) // 2],
                thickness=-1,
            )

    def simulate(self) -> None:
        count = 0
        speed = self.size * 100
        while True:
            self.transit()
            count += 1
            if count % speed != 0:
                continue
            padding = 20
            image = np.full(
                shape=(22 * self.size + 4 * padding, 22 * self.size + 2 * padding, 3),
                fill_value=255,
                dtype="uint8",
            )
            self.draw(image, padding)
            cv2.putText(
                image,
                f"tempature: {int(self.tempature * 100 + 0.5) / 100}",
                (0, 22 * self.size + 2 * padding),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=2,
                color=(0, 0, 0),
                thickness=2,
            )
            cv2.putText(
                image,
                f"magnetic: {int(self.magnetic * 100 + 0.5) / 100}",
                (0, 22 * self.size + padding),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=2,
                color=(0, 0, 0),
                thickness=2,
            )
            cv2.imshow("ising", image)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
            elif key == ord("p"):
                key = cv2.waitKey()
                if key == ord("q"):
                    break
            if key == ord("j"):
                self.tempature = max(self.tempature - 0.1, 0.0)
            if key == ord("k"):
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
