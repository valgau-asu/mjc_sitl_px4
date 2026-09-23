"""The controller interface.

A controller is any object with a `control(t, state) -> 4 rotor thrusts` method.
Subclassing `Controller` is optional but gives you the trajectory plumbing for free.
"""
import numpy as np


class Controller:
    """Base class: holds a trajectory and reports it back for scoring."""

    name = "controller"

    def __init__(self, traj, name=None):
        self.traj = traj
        if name:
            self.name = name

    def reset(self):
        """Called once before each run. Clear any internal state here."""

    def control(self, t, state):
        """Return four rotor thrusts in newtons, each within [0, MAX_THRUST]."""
        raise NotImplementedError

    def reference(self, t):
        """Where the controller was trying to be; used by the plots and the score."""
        return self.traj(t)[0]
