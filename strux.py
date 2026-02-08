import dataclasses
import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PyTree, PRNGKeyArray


def struct(Class=None, /, static_fieldnames=()):
    """
    Wrapper that transforms a class into an immutable dataclass that is also
    registered as a JAX PyTree.
    """
    if Class is None:
        return functools.partial(struct, static_fieldnames=static_fieldnames)
    
    # wrap class as an immutable Python dataclass
    Dataclass = dataclasses.dataclass(Class, frozen=True)
    
    # decide which fields are data vs. static
    fields = [field.name for field in dataclasses.fields(Dataclass)]
    data_fields = [name for name in fields if name not in static_fieldnames]
    meta_fields = [name for name in fields if name in static_fieldnames]
    # TODO: it should be an error to have invalid static fieldnames

    # register dataclass as a JAX pytree node
    jax.tree_util.register_dataclass(
        nodetype=Dataclass,
        data_fields=data_fields,
        meta_fields=meta_fields,
    )
    
    # overwrite string render methods to use pretty printing
    if "__repr__" not in Class.__dict__:
        Dataclass.__repr__ = to_str
    if "__str__" not in Class.__dict__:
        Dataclass.__str__ = to_str
    if "__format__" not in Class.__dict__:
        Dataclass.__format__ = format

    # other convenience methods
    Dataclass.replace = dataclasses.replace

    # allow type indexing
    Dataclass.__class_getitem__ = classmethod(lambda cls, _: cls)

    return Dataclass


def to_str(
    tree: PyTree,
    indent="  ",
    max_depth: int | None = None,
) -> str:
    lines = []
    def _put(s: str, depth: int):
        lines.append(indent * depth + s)
    def _walk(tree: PyTree, prefix: str, suffix: str, depth: int):
        if dataclasses.is_dataclass(tree):
            if depth == max_depth:
                _put(f"{prefix}{type(tree).__name__}(...){suffix}", depth=depth)
            else:
                _put(f"{prefix}{type(tree).__name__}(", depth=depth)
                state = tree.__getstate__() or {}
                for field, value in state.items():
                    _walk(value, prefix=f"{field}=", suffix=",", depth=depth+1)
                _put(f"){suffix}", depth=depth)
        elif isinstance(tree, tuple):
            if depth == max_depth:
                _put(f"{prefix}(...){suffix}", depth=depth)
            else:
                _put(f"{prefix}(", depth=depth)
                for item in tree:
                    _walk(item, prefix="", suffix=",", depth=depth+1)
                _put(f"){suffix}", depth=depth)
        elif isinstance(tree, list):
            if depth == max_depth:
                _put(f"{prefix}[...]{suffix}", depth=depth)
            else:
                _put(f"{prefix}[", depth=depth)
                for item in tree:
                    _walk(item, prefix="", suffix=",", depth=depth+1)
                _put(f"]{suffix}", depth=depth)
        elif isinstance(tree, dict):
            if depth == max_depth:
                _put(f"{prefix}{{...}}{suffix}", depth=depth)
            else:
                _put(f"{prefix}{{", depth=depth)
                for key, value in tree.items():
                    _walk(value, prefix=f"{key!r}: ", suffix=",", depth=depth+1)
                _put(f"}}{suffix}", depth=depth)
        elif isinstance(tree, np.ndarray):
            dtype = tree.dtype.name
            shape = str(tree.shape).strip("()").replace(",","")
            _put(f"{prefix}np.{dtype}[{shape}]{suffix}", depth=depth)
        elif isinstance(tree, jnp.ndarray):
            dtype = tree.dtype.name
            shape = str(tree.shape).strip("(,)").replace(" ","")
            _put(f"{prefix}jnp.{dtype}[{shape}]{suffix}", depth=depth)
        elif callable(tree):
            _put(f"{prefix}<fn:{tree.__name__}>{suffix}", depth=depth)
        elif isinstance(tree, (bool, int, float, str)):
            _put(f"{prefix}{type(tree).__name__}({tree!r}){suffix}", depth=depth)
        elif tree is None:
            _put(f"{prefix}None{suffix}", depth=depth)
        else:
            _put(f"{prefix}UNKNOWN_LEAF:{type(tree)}{suffix}", depth=depth)
    _walk(tree, prefix="", suffix="", depth=0)
    return "\n".join(lines)


def format(tree: PyTree, format_spec: str) -> str:
    # TODO: Do this properly with Python primitives for format string parsing
    # parse format spec
    if '.' in format_spec:
        max_depth_str, indent_size_str = format_spec.split('.')
    else:
        max_depth_str = format_spec
        indent_size_str = ""
    # parse max_depth_str
    max_depth = int(max_depth_str) if max_depth_str else None
    indent_size = int(indent_size_str) if indent_size_str else 2
    return to_str(tree, indent=" " * indent_size, max_depth=max_depth)


def split_treelike(
    key: PRNGKeyArray,
    tree: PyTree[Any, "S"],
) -> PyTree[PRNGKeyArray, "S"]:
    treedef = jax.tree.structure(tree)
    keys = jax.random.split(key, treedef.num_leaves)
    return jax.tree.unflatten(treedef, keys)


def size(tree: PyTree) -> int:
    """Calculates the total number of parameters in the PyTree."""
    return sum(jnp.size(x) for x in jax.tree.leaves(tree))
