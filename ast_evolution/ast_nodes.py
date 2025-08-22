"""
ast_evolution/ast_nodes.py - Expression tree nodes and safe primitives
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union
import random
import math

class ASTNode(ABC):
    """Base class for all AST nodes"""
    
    def __init__(self):
        self.arity = 0  # Number of children
        
    @abstractmethod
    def evaluate(self, x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        """Evaluate the node given input coordinates and time"""
        pass
        
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        pass
        
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ASTNode':
        """Deserialize from dictionary"""
        pass
        
    @abstractmethod
    def copy(self) -> 'ASTNode':
        """Create a deep copy of this node"""
        pass
        
    def get_all_nodes(self) -> List['ASTNode']:
        """Get all nodes in this subtree"""
        nodes = [self]
        if hasattr(self, 'children'):
            for child in self.children:
                nodes.extend(child.get_all_nodes())
        return nodes
        
    def get_depth(self) -> int:
        """Get maximum depth of this subtree"""
        if not hasattr(self, 'children'):
            return 1
        if not self.children:
            return 1
        return 1 + max(child.get_depth() for child in self.children)

class Variable(ASTNode):
    """Input variables: x, y, t"""
    
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.arity = 0
        
    def evaluate(self, x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        if self.name == 'x':
            return x
        elif self.name == 'y':
            return y
        elif self.name == 't':
            return np.full_like(x, t)
        else:
            return np.zeros_like(x)
            
    def to_dict(self) -> Dict[str, Any]:
        return {'type': 'Variable', 'name': self.name}
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Variable':
        return cls(data['name'])
        
    def copy(self) -> 'Variable':
        return Variable(self.name)
        
    def __str__(self):
        return self.name

class Constant(ASTNode):
    """Numeric constant"""
    
    def __init__(self, value: float):
        super().__init__()
        self.value = value
        self.arity = 0
        
    def evaluate(self, x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        return np.full_like(x, self.value)
        
    def to_dict(self) -> Dict[str, Any]:
        return {'type': 'Constant', 'value': self.value}
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Constant':
        return cls(data['value'])
        
    def copy(self) -> 'Constant':
        return Constant(self.value)
        
    def __str__(self):
        return f"{self.value:.3f}"

class UnaryOp(ASTNode):
    """Unary operations: sin, cos, tanh, abs, sqrt"""
    
    def __init__(self, op: str, child: ASTNode):
        super().__init__()
        self.op = op
        self.child = child
        self.children = [child]
        self.arity = 1
        
    def evaluate(self, x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        child_val = self.child.evaluate(x, y, t)
        
        # Safe numeric operations with fallbacks
        if self.op == 'sin':
            return np.sin(np.clip(child_val, -100, 100))
        elif self.op == 'cos':
            return np.cos(np.clip(child_val, -100, 100))
        elif self.op == 'tanh':
            return np.tanh(np.clip(child_val, -100, 100))
        elif self.op == 'abs':
            return np.abs(child_val)
        elif self.op == 'sqrt':
            return np.sqrt(np.maximum(np.abs(child_val), 1e-10))
        elif self.op == 'neg':
            return -child_val
        else:
            return child_val
            
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'UnaryOp',
            'op': self.op,
            'child': self.child.to_dict()
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnaryOp':
        child = node_from_dict(data['child'])
        return cls(data['op'], child)
        
    def copy(self) -> 'UnaryOp':
        return UnaryOp(self.op, self.child.copy())
        
    def __str__(self):
        return f"{self.op}({self.child})"

class BinaryOp(ASTNode):
    """Binary operations: add, mul, div with guards"""
    
    def __init__(self, op: str, left: ASTNode, right: ASTNode):
        super().__init__()
        self.op = op
        self.left = left
        self.right = right
        self.children = [left, right]
        self.arity = 2
        
    def evaluate(self, x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        left_val = self.left.evaluate(x, y, t)
        right_val = self.right.evaluate(x, y, t)
        
        # Safe numeric operations
        if self.op == 'add':
            result = left_val + right_val
        elif self.op == 'sub':
            result = left_val - right_val
        elif self.op == 'mul':
            result = left_val * right_val
        elif self.op == 'div':
            # Protected division
            divisor = np.where(np.abs(right_val) < 1e-10, 1.0, right_val)
            result = left_val / divisor
        elif self.op == 'mod':
            # Protected modulo
            divisor = np.where(np.abs(right_val) < 1e-10, 1.0, right_val)
            result = np.mod(left_val, divisor)
        elif self.op == 'pow':
            # Protected power with clipping
            base = np.clip(left_val, -100, 100)
            exp = np.clip(right_val, -10, 10)
            result = np.power(np.abs(base), exp) * np.sign(base)
        else:
            result = left_val
            
        # Clip to prevent overflow
        return np.clip(result, -1000, 1000)
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'BinaryOp',
            'op': self.op,
            'left': self.left.to_dict(),
            'right': self.right.to_dict()
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BinaryOp':
        left = node_from_dict(data['left'])
        right = node_from_dict(data['right'])
        return cls(data['op'], left, right)
        
    def copy(self) -> 'BinaryOp':
        return BinaryOp(self.op, self.left.copy(), self.right.copy())
        
    def __str__(self):
        return f"({self.left} {self.op} {self.right})"

class NoiseOp(ASTNode):
    """Perlin-like noise function"""
    
    def __init__(self, child: ASTNode, seed: int = None):
        super().__init__()
        self.child = child
        self.children = [child]
        self.arity = 1
        self.seed = seed or random.randint(0, 1000000)
        
    def evaluate(self, x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        scale = self.child.evaluate(x, y, t)
        
        # Simple noise using sin/cos with multiple frequencies
        np.random.seed(self.seed)
        freq1, freq2, freq3 = np.random.random(3) * 10 + 1
        
        noise = (np.sin(x * freq1 * scale + t) * np.cos(y * freq2 * scale + t) + 
                np.sin(x * freq2 * scale) * np.sin(y * freq3 * scale) +
                np.cos(x * freq3 * scale + t * 0.1)) / 3.0
                
        return noise
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'NoiseOp',
            'child': self.child.to_dict(),
            'seed': self.seed
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NoiseOp':
        child = node_from_dict(data['child'])
        return cls(child, data['seed'])
        
    def copy(self) -> 'NoiseOp':
        return NoiseOp(self.child.copy(), self.seed)
        
    def __str__(self):
        return f"noise({self.child})"

# Node creation helpers
def node_from_dict(data: Dict[str, Any]) -> ASTNode:
    """Create node from dictionary representation"""
    node_type = data['type']
    
    if node_type == 'Variable':
        return Variable.from_dict(data)
    elif node_type == 'Constant':
        return Constant.from_dict(data)
    elif node_type == 'UnaryOp':
        return UnaryOp.from_dict(data)
    elif node_type == 'BinaryOp':
        return BinaryOp.from_dict(data)
    elif node_type == 'NoiseOp':
        return NoiseOp.from_dict(data)
    else:
        raise ValueError(f"Unknown node type: {node_type}")

# Primitive sets for random generation
VARIABLES = ['x', 'y', 't']
UNARY_OPS = ['sin', 'cos', 'tanh', 'abs', 'sqrt', 'neg']
BINARY_OPS = ['add', 'sub', 'mul', 'div', 'mod', 'pow']

def create_random_node(depth: int = 0, max_depth: int = 5) -> ASTNode:
    """Create a random AST node"""
    if depth >= max_depth or random.random() < 0.3:
        # Terminal node
        if random.random() < 0.7:
            return Variable(random.choice(VARIABLES))
        else:
            return Constant(random.uniform(-2, 2))
    else:
        # Non-terminal node
        choice = random.random()
        if choice < 0.4:
            op = random.choice(UNARY_OPS)
            child = create_random_node(depth + 1, max_depth)
            return UnaryOp(op, child)
        elif choice < 0.8:
            op = random.choice(BINARY_OPS)
            left = create_random_node(depth + 1, max_depth)
            right = create_random_node(depth + 1, max_depth)
            return BinaryOp(op, left, right)
        else:
            child = create_random_node(depth + 1, max_depth)
            return NoiseOp(child)