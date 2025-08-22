"""
ast_evolution - Self-evolving generative algorithms using AST representation

A genetic programming system where genomes are expression trees that map
coordinates (x, y, t) to RGB values or audio samples.
"""

__version__ = "0.1.0"
__author__ = "AST Evolution Project"

from .genome import Genome
from .ast_nodes import (
    ASTNode, Variable, Constant, UnaryOp, BinaryOp, NoiseOp,
    create_random_node, node_from_dict,
    VARIABLES, UNARY_OPS, BINARY_OPS
)
from .population import Population
from .fitness import FitnessEvaluator, NoveltySearch, AestheticEvaluator
from .evaluator import Evaluator
from .archive import EvolutionArchive

__all__ = [
    'Genome',
    'ASTNode', 'Variable', 'Constant', 'UnaryOp', 'BinaryOp', 'NoiseOp',
    'create_random_node', 'node_from_dict',
    'VARIABLES', 'UNARY_OPS', 'BINARY_OPS',
    'Population',
    'FitnessEvaluator', 'NoveltySearch', 'AestheticEvaluator', 
    'Evaluator',
    'EvolutionArchive'
]