from queue import Queue, PriorityQueue
from dataclasses import dataclass

from utils.puzzle import Puzzle
from utils.state import State

from typing import *


@dataclass
class Solution():
    state: State
    heuristic: tuple
    visited: int
    queue: int


class ISAPuzzleSolver(Iterator[Solution]):

    
    def __init__(
        self, puzzle: Puzzle, *,
        states_limit: int = 10_000,
        heuristic: Callable[[State], tuple] | None = None,
    ) -> None:
        self.puzzle: Puzzle = puzzle
        self.heuristic: Callable[[State], tuple]  = heuristic or self.coherence_heuristic
        self.states_limit: int = states_limit

        self.queue: Queue[tuple[tuple, State]] = PriorityQueue()
        self.visited: set[State] = set()

        self.__generator: Iterator[Solution] = self.__search()
    
    
    def __next__(self) -> Solution:
        return next(self.__generator)
    

    def __iter__(self) -> Iterator[Solution]:
        return self
    

    def __search(self) -> Iterator[Solution]:
        initial_state = State.create_initial_state(self.puzzle)

        self.queue.put((self.heuristic(initial_state), initial_state))
        self.visited.add(initial_state)

        del initial_state

        while not self.queue.empty():
            score, state = self.queue.get()

            if state.is_complete():
                yield Solution(state, score, len(self.visited), self.queue.qsize())
            
            for action in state.actions:
                child = state.apply(*action)
                if child in self.visited: continue
                if len(self.visited) >= self.states_limit: break

                self.queue.put((self.heuristic(child), child), block=False)
                self.visited.add(child)
    

    @staticmethod
    def coherence_heuristic(state: State) -> tuple:
        return (state.coherence + len(state.available_cells) * 1000,)
    