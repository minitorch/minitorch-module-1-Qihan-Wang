"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    return x * y


def id(x: float) -> float:
    return x


def add(x: float, y: float) -> float:
    return x + y


def neg(x: float) -> float:
    return float(-x)


def lt(x: float, y: float) -> bool:
    return x < y


def eq(x: float, y: float) -> bool:
    return x == y


def max(x: float, y: float) -> float:
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    return abs(x - y) < 0.01


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    return x if x >= 0 else 0.0


def log(x: float) -> float:
    return math.log(x)


def exp(x: float) -> float:
    return math.exp(x)


def inv(x: float) -> float:
    return 1 / x


def log_back(x: float, y: float) -> float:
    return y / x


def inv_back(x: float, y: float) -> float:
    return -y / (x * x)


def relu_back(x: float, y: float) -> float:
    return y if x >= 0 else 0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(func: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    return lambda x: [func(i) for i in x]


def zipWith(
    func: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    return lambda x, y: [func(a, b) for a, b in zip(x, y)]


def reduce(
    func: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    def reduce_func(
        x: Iterable[float], func: Callable[[float, float], float], start: float
    ) -> float:
        result = start
        for elem in x:
            result = func(result, elem)
        return result

    return lambda x: reduce_func(x, func, start)


def negList(x: Iterable[float]) -> Iterable[float]:
    return map(neg)(x)


def addLists(x: Iterable[float], y: Iterable[float]) -> Iterable[float]:
    return zipWith(add)(x, y)


def sum(x: Iterable[float]) -> float:
    return reduce(add, 0)(x)


def prod(x: Iterable[float]) -> float:
    return reduce(mul, 1)(x)
