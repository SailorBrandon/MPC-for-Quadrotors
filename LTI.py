import imp
from pygame import init


import numpy as np

class LTI:
    def __init__(self, g, mass, Ixx, Iyy, Izz) -> None:
        self.Ac = None