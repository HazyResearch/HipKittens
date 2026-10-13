"""The exception a kernel author sees when hk refuses a kernel."""


class VerifyError(Exception):
    """The kernel is wrong or unbuildable. Never downgrade one of these."""


__all__ = ["VerifyError"]
