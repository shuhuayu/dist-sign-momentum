# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Any, Callable, Optional, TypeVar

# Used for annotating the decorator usage of 'no_grad' and 'enable_grad'.
# See https://mypy.readthedocs.io/en/latest/generics.html#declaring-decorators
FuncType = Callable[..., Any]
T = TypeVar('T', bound=FuncType)

class no_grad:
    def __enter__(self) -> None: ...
    def __exit__(self, *args: Any) -> Optional[bool]: ...
    def __call__(self, func: T) -> T: ...

class enable_grad:
    def __enter__(self) -> None: ...
    def __exit__(self, *args: Any) -> Optional[bool]: ...
    def __call__(self, func: T) -> T: ...

class set_grad_enabled:
    def __init__(self, mode: bool) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, *args: Any) -> Optional[bool]: ...
