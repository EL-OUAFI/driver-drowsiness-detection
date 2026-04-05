"""Project-specific exceptions."""


class ProjectError(Exception):
    """Base class for repository-specific errors."""


class MissingDependencyError(ProjectError):
    """Raised when an optional runtime dependency is unavailable."""


class DatasetStructureError(ProjectError):
    """Raised when the dataset layout cannot be interpreted safely."""


class FaceNotFoundError(ProjectError):
    """Raised when no face can be localized in a frame."""
