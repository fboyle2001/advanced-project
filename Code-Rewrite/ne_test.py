from typing import Dict, List, Tuple, Generic, TypeVar, Iterator, Any

import bisect
import pickle

T = TypeVar("T")

class SingleClassLimitedPriorityBuffer(Generic[T]):
    def __init__(self, max_length: int, high_priority: bool = True):
        self.max_length = max_length
        self.high_priority = high_priority

        # Order: Lowest priority --> Highest priority
        # Search by priority: identifiy index i in self.priorities -> draw self.hash_map[self.ordered_hashes[i]] as sample
        self.priorities: List[float] = []
        self.ordered_hashes: List[int] = []
        self.hash_map: Dict[int, T] = {}

    def __len__(self) -> int:
        return len(self.priorities)

    def __iter__(self) -> Iterator[Tuple[float, T]]:
        for priority, hashed in zip(self.priorities, self.ordered_hashes):
            yield priority if not self.high_priority else -priority, self.hash_map[hashed]

    def __str__(self) -> str:
        return f"[{', '.join([str(x) for x in iter(self)])}]"

    def is_full(self) -> bool:
        return len(self) == self.max_length

    def add_sample(self, item: T, priority: float) -> None:
        if self.high_priority:
            priority *= -1

        # If full and priority is lower than the lowest priority, ignore it
        if self.is_full() and priority >= self.priorities[-1]:
            return

        # Calculate the hash key
        hashed = hash(pickle.dumps(item))
        self.hash_map[hashed] = item

        # Insert into the arrays in sorted position
        insert_idx = bisect.bisect(self.priorities, priority)
        self.priorities.insert(insert_idx, priority)
        self.ordered_hashes.insert(insert_idx, hashed)

        self._remove_one_sample()

    def _remove_one_sample(self) -> None:
        # No need to remove a sample if there is still space
        if len(self) <= self.max_length:
            return
        
        # Remove the hashed value
        delete_hash = self.ordered_hashes[-1]
        del self.hash_map[delete_hash]
        
        # Remove one sample
        old_length = len(self.priorities)
        self.priorities = self.priorities[:old_length - 1]
        self.ordered_hashes = self.ordered_hashes[:old_length - 1]

    def reduce_max_length(self, new_max_length: int) -> None:
        if self.max_length < new_max_length:
            self.max_length = new_max_length
            return

        self.max_length = new_max_length

        while len(self) > self.max_length:
            self._remove_one_sample()

class LimitedPriorityBuffer(Generic[T]):
    def __init__(self, max_samples: int, high_priority: bool = True):
        self.max_samples = max_samples
        self.high_priority = high_priority

        self.target_buffers: Dict[Any, SingleClassLimitedPriorityBuffer[T]] = dict()
    
    @property
    def known_targets(self) -> List[Any]:
        return list(self.target_buffers.keys())
    
    def __len__(self):
        return sum([len(buffer) for buffer in self.target_buffers.values()])

    def __str__(self):
        return str({str(target): str(self.target_buffers[target]) for target in self.known_targets})

    def add_sample(self, target: Any, item: T, priority: float) -> None:
        # Add the new target if it does not exist
        if target not in self.target_buffers.keys():
            self._add_new_target(target)
        
        # Add the sample to the correct buffer
        self.target_buffers[target].add_sample(item, priority)

    def _add_new_target(self, target: Any):
        # Ignore if it is already in the buffer
        if target in self.target_buffers.keys():
            return

        # Recalculate the max samples per class
        max_samples_per_class = self.max_samples // (len(self.known_targets) + 1)
        self.target_buffers[target] = SingleClassLimitedPriorityBuffer(max_samples_per_class, high_priority=self.high_priority)

        # Reduce the size of existing buffers accordingly w.r.t to the priorities
        for known_target in self.known_targets:
            self.target_buffers[known_target].reduce_max_length(max_samples_per_class)

# test = SingleClassLimitedPriorityBuffer[str](5, high_priority=True)

# test.add_sample("a", 3)
# test.add_sample("b", 1)
# test.add_sample("c", 4)
# test.add_sample("d", 6)
# test.add_sample("e", 5)
# print(test)
# test.add_sample("f", 2)
# print(test)

# c = LimitedPriorityBuffer[str](4, high_priority=True)
# c.add_sample("dog", "a", 3)
# c.add_sample("dog", "b", 1)
# c.add_sample("dog", "c", 4)
# c.add_sample("dog", "d", 2)
# print(c)
# c.add_sample("cat", "x", 5)
# c.add_sample("dog", "m", 10)
# print(c)