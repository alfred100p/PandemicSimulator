from gym.utils import seeding


class Space(object):
    """Defines the observation and action spaces, so you can write generic
    code that applies to any Env. For example, you can choose a random
    action.
    WARNING - Custom observation & action spaces can inherit from the `Space`
    class. However, most use-cases should be covered by the existing space
    classes (e.g. `Box`, `Discrete`, etc...), and container classes (`Tuple` &
    `Dict`). Note that parametrized probability distributions (through the
    `sample()` method), and batching functions (in `gym.vector.VectorEnv`), are
    only well-defined for instances of spaces provided in gym by default.
    Moreover, some implementations of Reinforcement Learning algorithms might
    not handle custom spaces properly. Use custom spaces with care.
    """

    def __init__(self, shape=None, dtype=None, seed=None):
        import numpy as np  # takes about 300-400ms to import, so we load lazily

        self._shape = None if shape is None else tuple(shape)
        self.dtype = None if dtype is None else np.dtype(dtype)
        self._np_random = None
        if seed is not None:
            self.seed(seed)

    @property
    def np_random(self):
        """Lazily seed the rng since this is expensive and only needed if
        sampling from this space.
        """
        if self._np_random is None:
            self.seed()

        return self._np_random

    @property
    def shape(self):
        """Return the shape of the space as an immutable property"""
        return self._shape

    def sample(self):
        """Randomly sample an element of this space. Can be
        uniform or non-uniform sampling based on boundedness of space."""
        raise NotImplementedError

    def seed(self, seed=None):
        """Seed the PRNG of this space."""
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    def contains(self, x):
        """
        Return boolean specifying if x is a valid
        member of this space
        """
        raise NotImplementedError

    def __contains__(self, x):
        return self.contains(x)

    def __setstate__(self, state):
        # Don't mutate the original state
        state = dict(state)

        # Allow for loading of legacy states.
        # See:
        #   https://github.com/openai/gym/pull/2397 -- shape
        #   https://github.com/openai/gym/pull/1913 -- np_random
        #
        if "shape" in state:
            state["_shape"] = state["shape"]
            del state["shape"]
        if "np_random" in state:
            state["_np_random"] = state["np_random"]
            del state["np_random"]

        # Update our state
        self.__dict__.update(state)

    def to_jsonable(self, sample_n):
        """Convert a batch of samples from this space to a JSONable data type."""
        # By default, assume identity is JSONable
        return sample_n

    def from_jsonable(self, sample_n):
        """Convert a JSONable data type to a batch of samples from this space."""
        # By default, assume identity is JSONable
        return sample_n
class Discrete(Space):
    r"""A discrete space in :math:`\{ 0, 1, \\dots, n-1 \}`.
    A start value can be optionally specified to shift the range
    to :math:`\{ a, a+1, \\dots, a+n-1 \}`.
    Example::
        >>> Discrete(2)
        >>> Discrete(3, start=-1)  # {-1, 0, 1}
    """

    def __init__(self, n, seed=None, start=0):
        assert n >= 0 and isinstance(start, (int, np.integer))
        self.n = n
        self.start = int(start)
        super(Discrete, self).__init__((), np.int64, seed)

    def sample(self):
        return self.start + self.np_random.randint(self.n)

    def contains(self, x):
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and (
            x.dtype.char in np.typecodes["AllInteger"] and x.shape == ()
        ):
            as_int = int(x)
        else:
            return False
        return self.start <= as_int < self.start + self.n

    def __repr__(self):
        if self.start != 0:
            return "Discrete(%d, start=%d)" % (self.n, self.start)
        return "Discrete(%d)" % self.n

    def __eq__(self, other):
        return (
            isinstance(other, Discrete)
            and self.n == other.n
            and self.start == other.start
        )
