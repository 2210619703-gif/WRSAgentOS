"""Robot action parameters and contracts; implementations belong to the chosen environment."""

from importlib.resources import files
from typing import Literal

from pydantic import Field, model_validator

from wrs_agent.schemas import Boundary, Empty, Name
from wrs_agent.skills.contracts import Skill


class MoveArgs(Boundary):
    pose: Literal["home", "B", "C"]


class RelativeMoveArgs(Boundary):
    dx: float = Field(default=0.0, ge=-0.1, le=0.1, description="World X displacement in meters")
    dy: float = Field(default=0.0, ge=-0.1, le=0.1, description="World Y displacement in meters")
    dz: float = Field(default=0.0, ge=-0.1, le=0.1, description="World Z displacement in meters")

    @model_validator(mode="after")
    def bounded_displacement(self):
        distance_squared = self.dx**2 + self.dy**2 + self.dz**2
        if not 0 < distance_squared <= 0.1**2 + 1e-12:
            raise ValueError("displacement_must_be_nonzero_and_at_most_10cm")
        return self


class GripperArgs(Boundary):
    command: Literal["open", "close"]


class PickArgs(Boundary):
    object: Name


class PlaceArgs(PickArgs):
    target: Literal["B", "C"]


_GUIDE = files(__package__).joinpath("SKILL.md").read_text(encoding="utf-8")


def _skill(
    name,
    arguments,
    description,
    *,
    preconditions=(),
    recovery=(),
    verification="virtual_world_postcondition",
):
    return Skill(
        name=name,
        description=description,
        instructions=_GUIDE,
        resources=("arm",),
        preconditions=preconditions,
        recovery=recovery,
        verification=verification,
        arguments=arguments,
    )


SKILLS = {
    entry.name: entry
    for entry in (
        _skill(
            "observe",
            Empty,
            "Refresh robot and environment state only; no camera detection and no text output",
        ),
        _skill(
            "move_named_pose",
            MoveArgs,
            "Move to a validated named joint pose",
            preconditions=("known_pose",),
        ),
        _skill(
            "move_relative",
            RelativeMoveArgs,
            "Offset the flange in world XYZ meters at most 10 cm: up/down is ±Z, "
            "left/right is ±Y, forward/back is ±X; preserve endpoint orientation, "
            "joint-interpolated path without collision checking",
            preconditions=("reachable_target",),
            verification="wrs_fk",
        ),
        _skill(
            "set_gripper",
            GripperArgs,
            "Fully open or fully close the parallel gripper jaw; the arm does not move. "
            "Closing is jaw motion only: it never confirms contact or holding an object",
            verification="wrs_fk",
        ),
        _skill(
            "pick",
            PickArgs,
            "Acquire an observed object and verify holding",
            preconditions=("empty_gripper", "object_visible"),
            recovery=("observe_once",),
        ),
        _skill(
            "place",
            PlaceArgs,
            "Place the held object at a known target",
            preconditions=("object_held", "known_target"),
        ),
        _skill(
            "verify",
            PlaceArgs,
            "Confirm object placement and empty gripper",
        ),
    )
}
