from typing import Union, List, Tuple, Optional, Any, Literal
import numpy as np


class QuantumGate:
    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = matrix

    @staticmethod
    def X() -> 'QuantumGate':
        return QuantumGate(np.array([[0,1],[1,0]], dtype=np.complex64))

    @staticmethod
    def Y() -> 'QuantumGate':
        return QuantumGate(np.array([[0,-1j],[1j,0]], dtype=np.complex64))

    @staticmethod
    def Z() -> 'QuantumGate':
        return QuantumGate(np.array([[1,0],[0,-1]], dtype=np.complex64))

    @staticmethod
    def H() -> 'QuantumGate':
        return QuantumGate(1/np.sqrt(2) * np.array([[1,1],[1,-1]], dtype=np.complex64))

    @staticmethod
    def Rx(theta: float) -> 'QuantumGate':
        c = np.cos(theta/2)
        s = -1j*np.sin(theta/2)
        return QuantumGate(np.array([[c, s],[s, c]], dtype=np.complex64))

    @staticmethod
    def Ry(theta: float) -> 'QuantumGate':
        c = np.cos(theta/2)
        s = np.sin(theta/2)
        return QuantumGate(np.array([[c, -s],[s, c]], dtype=np.complex64))

    @staticmethod
    def Rz(theta: float) -> 'QuantumGate':
        return QuantumGate(np.array([[np.exp(-1j*theta/2), 0],[0, np.exp(1j*theta/2)]], dtype=np.complex64))

    @staticmethod
    def T() -> 'QuantumGate':
        return QuantumGate(np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex64))

    @staticmethod
    def S() -> 'QuantumGate':
        return QuantumGate(np.array([[1, 0], [0, 1j]], dtype=np.complex64))

    @staticmethod
    def CNOT() -> 'QuantumGate':
        return QuantumGate(np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=np.complex64))

    @staticmethod
    def Toffoli() -> 'QuantumGate':
        return QuantumGate(np.array([[1,0,0,0,0,0,0,0],
                                    [0,1,0,0,0,0,0,0],
                                    [0,0,1,0,0,0,0,0],
                                    [0,0,0,1,0,0,0,0],
                                    [0,0,0,0,1,0,0,0],
                                    [0,0,0,0,0,1,0,0],
                                    [0,0,0,0,0,0,0,1],
                                    [0,0,0,0,0,0,1,0]], dtype=np.complex64))

    @staticmethod
    def SWAP() -> 'QuantumGate':
        # 2-qubit SWAP gate (4x4)
        return QuantumGate(np.array([
            [1,0,0,0],
            [0,0,1,0],
            [0,1,0,0],
            [0,0,0,1]
        ], dtype=np.complex64))

    @staticmethod
    def CRZ(theta):
        # Controlled-Rz gate in 4x4 matrix form
        import numpy as np
        exp = np.exp
        half_theta = theta / 2
        return QuantumGate(np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, exp(-1j * half_theta), 0],
            [0, 0, 0, exp(1j * half_theta)]
        ], dtype=np.complex64))

    def isunitary(self) -> bool:
        return np.allclose(np.eye(self.matrix.shape[0]), self.matrix @ self.matrix.conj().T)

    def __mul__(self, other: Any) -> Any:
        # Matrix multiplication with another gate or state
        if isinstance(other, QuantumGate):
            return QuantumGate(np.dot(self.matrix, other.matrix))
        elif isinstance(other, QuantumState):
            return other.gate(self)
        else:
            raise ValueError("Can only multiply with QuantumGate or QuantumState")

    def __matmul__(self, other: 'QuantumGate') -> 'QuantumGate':
        # Tensor product of two gates call by using @ operator
        return QuantumGate(np.kron(self.matrix, other.matrix))

    def __repr__(self):
        return f"QuantumGate({self.matrix})"

    def apply_to(self, num_qubits: int, which: Union[int, List[int]]) -> 'QuantumGate':
        """
        Returns a new QuantumGate that acts as this gate on the specified qubit(s) in a num_qubits system.
        For single-qubit gates, which is int. For CNOT/Toffoli, which is list of ints.
        """
        eye2 = np.eye(2, dtype=np.complex64)
        if isinstance(which, int):
            # Single-qubit gate
            ops = []
            for i in range(num_qubits):
                if i == which:
                    ops.append(self.matrix)
                else:
                    ops.append(eye2)
            result = ops[0]
            for op in ops[1:]:
                result = np.kron(result, op)
            return QuantumGate(result)
        elif isinstance(which, list):
            # Multi-qubit gate (CNOT, Toffoli, SWAP, or controlled-1qubit)
            if self.matrix.shape == (2, 2) and len(which) == 2:
                ctrl, tgt = which
                return QuantumGate(QuantumGate._expand_controlled_gate(num_qubits, ctrl, tgt, self.matrix))
            elif self.matrix.shape[0] == 4 and len(which) == 2:
                # Could be CNOT or SWAP or CRZ
                if np.allclose(self.matrix, QuantumGate.CNOT().matrix):
                    ctrl, tgt = which
                    return QuantumGate(QuantumGate._expand_cnot(num_qubits, ctrl, tgt))
                elif np.allclose(self.matrix, QuantumGate.SWAP().matrix):
                    q1, q2 = which
                    return QuantumGate(QuantumGate._expand_swap(num_qubits, q1, q2))
                else:
                    # Assume controlled-1qubit gate (like CRZ)
                    ctrl, tgt = which
                    return QuantumGate(QuantumGate._expand_controlled_gate(num_qubits, ctrl, tgt, self.matrix))
            elif self.matrix.shape[0] == 8:  # Toffoli
                ctrl1, ctrl2, tgt = which
                return QuantumGate(QuantumGate._expand_toffoli(num_qubits, ctrl1, ctrl2, tgt))
            else:
                raise NotImplementedError("Multi-qubit gate expansion not implemented for this gate.")
        else:
            raise ValueError("which must be int or list of ints.")
    
    @staticmethod
    def expand_multi_controlled_gate(n: int, controls: List[int], target: int, base_gate: np.ndarray) -> np.ndarray:
        """
        Returns the matrix for a multi-controlled 1-qubit gate (base_gate) acting on qubits 'controls' (list of control indices)
        and 'target' (target index) in n qubits.
        base_gate must be a 2x2 matrix.
        """
        dim = 2 ** n
        result = np.zeros((dim, dim), dtype=np.complex64)
        for i in range(dim):
            bits = [(i >> k) & 1 for k in reversed(range(n))]
            if all(bits[c] == 1 for c in controls):
                # Apply base_gate to target qubit
                for j in range(2):
                    bits_copy = bits.copy()
                    bits_copy[target] = j
                    idx = 0
                    for b in bits_copy:
                        idx = (idx << 1) | b
                    result[idx, i] = base_gate[j, bits[target]]
            else:
                idx = 0
                for b in bits:
                    idx = (idx << 1) | b
                result[idx, i] = 1
        return result

    @staticmethod
    def _expand_cnot(n: int, ctrl: int, tgt: int) -> np.ndarray:
        """Returns the matrix for a CNOT gate acting on qubits ctrl (control) and tgt (target) in n qubits."""
        dim = 2 ** n
        result = np.zeros((dim, dim), dtype=np.complex64)
        for i in range(dim):
            bits = [(i >> k) & 1 for k in reversed(range(n))]
            if bits[ctrl] == 1:
                bits[tgt] ^= 1
            j = 0
            for b in bits:
                j = (j << 1) | b
            result[j, i] = 1
        return result
    
    @staticmethod
    def _expand_toffoli(n: int, ctrl1: int, ctrl2: int, tgt: int) -> np.ndarray:
        """Returns the matrix for a Toffoli gate acting on qubits ctrl1, ctrl2 (controls) and tgt (target) in n qubits."""
        dim = 2 ** n
        result = np.zeros((dim, dim), dtype=np.complex64)
        for i in range(dim):
            bits = [(i >> k) & 1 for k in reversed(range(n))]
            if bits[ctrl1] == 1 and bits[ctrl2] == 1:
                bits[tgt] ^= 1
            j = 0
            for b in bits:
                j = (j << 1) | b
            result[j, i] = 1
        return result
    
    @staticmethod
    def _expand_controlled_gate(n: int, ctrl: int, tgt: int, base_gate: np.ndarray) -> np.ndarray:
        """
        Returns the matrix for a controlled-1 qubit gate (base_gate) acting on qubits ctrl (control) and tgt (target) in n qubits.
        base_gate must be a 2x2 matrix.
        """
        dim = 2 ** n
        result = np.zeros((dim, dim), dtype=np.complex64)
        for i in range(dim):
            bits = [(i >> k) & 1 for k in reversed(range(n))]
            if bits[ctrl] == 1:
                # Apply base_gate to target qubit
                bits_copy = bits.copy()
                for j in range(2):
                    bits_copy[tgt] = j
                    idx = 0
                    for b in bits_copy:
                        idx = (idx << 1) | b
                    result[idx, i] = base_gate[j, bits[tgt]]
            else:
                idx = 0
                for b in bits:
                    idx = (idx << 1) | b
                result[idx, i] = 1
        return result

    @staticmethod
    def _expand_swap(n: int, q1: int, q2: int) -> np.ndarray:
        """
        Returns the matrix for a SWAP gate acting on qubits q1 and q2 in n qubits.
        """
        dim = 2 ** n
        result = np.zeros((dim, dim), dtype=np.complex64)
        for i in range(dim):
            bits = [(i >> k) & 1 for k in reversed(range(n))]
            bits_swapped = bits.copy()
            bits_swapped[q1], bits_swapped[q2] = bits_swapped[q2], bits_swapped[q1]
            j = 0
            for b in bits_swapped:
                j = (j << 1) | b
            result[j, i] = 1
        return result


class QuantumState:
    
    def __init__(self, vector: Optional[Union[List[complex], np.ndarray]] = None, basis: Optional[str] = None) -> None:
        if basis is not None:
            self.vector = self._basis_vector(basis)
        elif vector is not None:
            if isinstance(vector, list):
                vector = np.array(vector, dtype=np.complex64)
            self.vector = vector / np.linalg.norm(vector)
        else:
            raise ValueError("Must provide either vector or basis.")

    @staticmethod
    def _basis_vector(basis: str) -> np.ndarray:
        # Standard basis states for 1 qubit
        if basis == '0':
            return np.array([1,0], dtype=np.complex64)
        elif basis == '1':
            return np.array([0,1], dtype=np.complex64)
        elif basis == '+':
            return 1/np.sqrt(2) * np.array([1,1], dtype=np.complex64)
        elif basis == '-':
            return 1/np.sqrt(2) * np.array([1,-1], dtype=np.complex64)
        elif basis == 'i':
            return 1/np.sqrt(2) * np.array([1,1j], dtype=np.complex64)
        elif basis == '-i':
            return 1/np.sqrt(2) * np.array([1,-1j], dtype=np.complex64)
        else:
            raise ValueError(f"Unknown basis state: {basis}")

    @staticmethod
    def tensor(states: List['QuantumState']) -> 'QuantumState':
        """Tensor product of a list of QuantumState objects."""
        vec = states[0].vector
        for s in states[1:]:
            vec = np.kron(vec, s.vector)
        return QuantumState(vec)
    
    def gate(self, gate: QuantumGate, mutate: bool = False) -> Optional['QuantumState']:
        if gate.matrix.shape[1] != self.vector.shape[0]:
            raise ValueError(f"Gate and state dimension mismatch: gate {gate.matrix.shape}, state {self.vector.shape}")
        if not mutate:
            new_vector = np.dot(gate.matrix, self.vector)
            return QuantumState(new_vector)
        else:
            self.vector = gate.matrix @ self.vector
            self.vector /= np.linalg.norm(self.vector)

    def apply_gate_old(self, gate: QuantumGate, which: Union[int, List[int]], mutate: bool = False) -> Optional['QuantumState']:
        """
        Apply a gate to the specified qubit(s) in the multi-qubit state.
        which: int for single-qubit, list for multi-qubit gates.
        """
        n = int(np.log2(self.vector.shape[0]))
        expanded_gate = gate.apply_to(n, which)
        return self.gate(expanded_gate, mutate=mutate)
    
    def apply_gate(self, gate: QuantumGate, which: Union[int, List[int]], mutate: bool = False) -> Optional['QuantumState']:
        if isinstance(which, int):
                which = [which]
        which = list(which)
        
        n_qubits = int(np.log2(self.vector.size))
        k = len(which)
        if gate.matrix.shape != (2**k, 2**k):
            raise ValueError(f"Gate matrix shape {gate.matrix.shape} doesn't match {k}-qubit operation.")

        # Step 1: Reshape state vector to tensor
        state_tensor = self.vector.reshape([2] * n_qubits)

        # Step 2: Permute target qubits to front
        perm = which + [i for i in range(n_qubits) if i not in which]
        inv_perm = np.argsort(perm)
        permuted_state = np.transpose(state_tensor, axes=perm)

        # Step 3: Reshape for gate application
        permuted_state = permuted_state.reshape((2**k, -1))
        updated = gate.matrix @ permuted_state

        # Step 4: Reshape back and invert permutation
        updated = updated.reshape([2] * n_qubits)
        updated = np.transpose(updated, axes=inv_perm)
        updated_vector = updated.reshape(2**n_qubits)

        if mutate:
            self.vector = updated_vector / np.linalg.norm(updated_vector)
        else:
            return QuantumState(updated_vector)

    def measure(self, basis: Literal['computational','+/-', 'i/-i'] = 'computational', 
                return_probabilities: bool = False) -> Union[int, np.ndarray]:
        # Optionally measure in different bases
        v = self.vector
        if basis == 'computational':
            probs = np.abs(v) ** 2
        elif basis == '+/-':
            # Hadamard basis
            H = QuantumGate.H().matrix
            v = H @ v
            probs = np.abs(v) ** 2
        elif basis == 'i/-i':
            # Y basis
            Y = QuantumGate.Rx(np.pi/2).matrix
            v = Y @ v
            probs = np.abs(v) ** 2
        else:
            raise ValueError(f"Unknown measurement basis: {basis}")
        if return_probabilities:
            return probs
        outcomes = np.arange(len(probs))
        return np.random.choice(outcomes, p=probs)

    def density_matrix(self) -> np.ndarray:
        """Returns the density matrix of the state."""
        return np.outer(self.vector, self.vector.conj())

    def trace(self) -> complex:
        """Returns the trace of the density matrix (should be 1 for normalized states)."""
        rho = self.density_matrix()
        return np.trace(rho)

    def purity(self) -> float:
        """Returns Tr(rho^2), which is 1 for pure states, <1 for mixed."""
        rho = self.density_matrix()
        return np.real(np.trace(rho @ rho))

    def is_pure(self, tol: float = 1e-10) -> bool:
        """Checks if the state is pure (Tr(rho^2) == 1)."""
        return abs(self.purity() - 1.0) < tol

    def is_mixed(self, tol: float = 1e-10) -> bool:
        """Checks if the state is mixed (Tr(rho^2) < 1)."""
        return self.purity() < 1 - tol

    def __mul__(self, other: 'QuantumState') -> 'QuantumState':
        """Tensor product using * operator for convenience."""
        if not isinstance(other, QuantumState):
            raise ValueError("Can only multiply with another QuantumState")
        return QuantumState(np.kron(self.vector, other.vector))

    def __matmul__(self, other: 'QuantumState') -> 'QuantumState':
        # Tensor product of two states with @ operator
        return QuantumState(np.kron(self.vector, other.vector))

    def __repr__(self):
        # Return pretty string representation of the states density matrix
        return f"QuantumState({self.vector})"
        return f"QuantumState({self.vector})"
