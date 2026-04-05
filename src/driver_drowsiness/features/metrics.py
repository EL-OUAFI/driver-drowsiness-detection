"""Geometric feature computations independent of the vision stack."""

from __future__ import annotations

from math import dist
from typing import Sequence

Point = tuple[float, float]


def apply_displacements(points: Sequence[Point], offsets: Sequence[Point]) -> list[Point]:
    """Translate a set of points by point-wise offsets."""
    if len(points) != len(offsets):
        raise ValueError("points and offsets must have the same length.")
    return [(x + dx, y + dy) for (x, y), (dx, dy) in zip(points, offsets, strict=True)]


def bounding_box(points: Sequence[Point], padding: int = 0) -> tuple[int, int, int, int]:
    """Return an integer bounding box around a set of points."""
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (
        int(min(xs)) - padding,
        int(min(ys)) - padding,
        int(max(xs)) + padding,
        int(max(ys)) + padding,
    )


def eye_aspect_ratio(points: Sequence[Point]) -> float:
    """Compute the eye aspect ratio from six eye landmarks."""
    if len(points) != 6:
        raise ValueError("EAR expects exactly six eye landmarks.")
    horizontal = dist(points[0], points[3])
    if horizontal == 0:
        return 0.0
    vertical = dist(points[1], points[5]) + dist(points[2], points[4])
    return vertical / (2.0 * horizontal)


def mouth_aspect_ratio(points: Sequence[Point]) -> float:
    """Compute the mouth aspect ratio from the 12 outer-mouth landmarks."""
    if len(points) != 12:
        raise ValueError("MAR expects exactly twelve outer-mouth landmarks.")
    horizontal = dist(points[0], points[6])
    if horizontal == 0:
        return 0.0
    vertical = dist(points[2], points[10]) + dist(points[4], points[8])
    return vertical / (2.0 * horizontal)
