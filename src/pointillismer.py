import pickle

import cv2
import numpy as np


Point = tuple[int, int]


class Pointillismer:
    id: int = 0

    def __init__(self, name: str, shape: tuple[int, int, int] = (640, 640, 3)) -> None:
        self.id = self.__class__.id
        self.__class__.id += 1
        self.window_name = f"{name},id={self.id}"
        self.image = np.zeros(shape=shape, dtype=np.uint8)
        self.points: list[Point] = []
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.add_or_del_point)

    def add_or_del_point(self, event: int, x: int, y: int, *args) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            cv2.circle(self.image, (x, y), 5, (0, 255, 255), -1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            x, y = self.points.pop()
            cv2.circle(self.image, (x, y), 5, (0, 0, 0), -1)

    def start(self) -> None:
        while True:
            cv2.imshow(self.window_name, self.image)
            if cv2.waitKey(50) & 0xFF == 27:
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-f", "--filename", required=True)
    args = parser.parse_args()
    filename: str = f"{args.filename}.pkl"
    pl = Pointillismer("pointillismer")
    pl.start()

    points = pl.points

    with open(filename, "wb") as f:
        pickle.dump(points, f)
