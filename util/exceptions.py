class IntegrityError(Exception):
    def __init__(self, message: str = "Integrity constraint violated.") -> None:
        self.message = message
        super().__init__(message)
