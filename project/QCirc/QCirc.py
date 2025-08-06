from QGSTates import QuantumGate, QuantumState
import numpy as np
from typing import List, Optional, Union, Tuple


class QuantumCircuit:
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.gate_sequence: List[Tuple[QuantumGate, Union[int, List[int]]]] = []
        self.number_of_gates = 0
    def add_gate(self, gate: QuantumGate, which: Union[int, List[int]]):
        self.gate_sequence.append((gate, which))
        self.number_of_gates += 1

    def run(self, initial_state: Union[QuantumState, None] = None) -> QuantumState:
        if initial_state is None:
            state = QuantumState([1] + [0]*(2**self.num_qubits - 1))  # |0...0>
        else:
            state = initial_state

        for gate, which in self.gate_sequence:
            state = state.apply_gate(gate, which)
        return state