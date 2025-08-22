"""
ast_evolution/genome.py - Genome representation and JSON serialization
"""
import json
import random
from typing import Dict, Any, List, Tuple
from .ast_nodes import ASTNode, create_random_node, node_from_dict

class Genome:
    """Represents an individual as a collection of expression trees"""
    
    def __init__(self, trees: Dict[str, ASTNode] = None, mode: str = 'image'):
        self.mode = mode  # 'image' or 'audio'
        
        if trees is None:
            self.trees = self._create_random_trees()
        else:
            self.trees = trees
            
        self.fitness = 0.0
        self.novelty = 0.0
        self.aesthetic = 0.0
        self.age = 0
        
    def _create_random_trees(self) -> Dict[str, ASTNode]:
        """Create random expression trees based on mode"""
        trees = {}
        
        if self.mode == 'image':
            # RGB channels
            trees['r'] = create_random_node(max_depth=random.randint(3, 7))
            trees['g'] = create_random_node(max_depth=random.randint(3, 7))
            trees['b'] = create_random_node(max_depth=random.randint(3, 7))
        else:  # audio
            # Single audio sample output
            trees['audio'] = create_random_node(max_depth=random.randint(4, 8))
            
        return trees
        
    def evaluate(self, x, y, t) -> Dict[str, Any]:
        """Evaluate all trees for given inputs"""
        results = {}
        for channel, tree in self.trees.items():
            try:
                results[channel] = tree.evaluate(x, y, t)
            except Exception as e:
                # Fallback for evaluation errors
                print(f"Evaluation error in {channel}: {e}")
                results[channel] = x * 0  # Return zeros
        return results
        
    def get_complexity(self) -> int:
        """Get total complexity (number of nodes) across all trees"""
        return sum(len(tree.get_all_nodes()) for tree in self.trees.values())
        
    def get_depth(self) -> int:
        """Get maximum depth across all trees"""
        return max(tree.get_depth() for tree in self.trees.values())
        
    def copy(self) -> 'Genome':
        """Create a deep copy of this genome"""
        new_trees = {name: tree.copy() for name, tree in self.trees.items()}
        new_genome = Genome(new_trees, self.mode)
        new_genome.fitness = self.fitness
        new_genome.novelty = self.novelty
        new_genome.aesthetic = self.aesthetic
        new_genome.age = self.age
        return new_genome
        
    def to_dict(self) -> Dict[str, Any]:
        """Serialize genome to dictionary"""
        return {
            'mode': self.mode,
            'trees': {name: tree.to_dict() for name, tree in self.trees.items()},
            'fitness': self.fitness,
            'novelty': self.novelty,
            'aesthetic': self.aesthetic,
            'age': self.age,
            'complexity': self.get_complexity(),
            'depth': self.get_depth()
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Genome':
        """Deserialize genome from dictionary"""
        trees = {name: node_from_dict(tree_data) 
                for name, tree_data in data['trees'].items()}
        genome = cls(trees, data.get('mode', 'image'))
        genome.fitness = data.get('fitness', 0.0)
        genome.novelty = data.get('novelty', 0.0)
        genome.aesthetic = data.get('aesthetic', 0.0)
        genome.age = data.get('age', 0)
        return genome
        
    def to_json(self, filename: str = None) -> str:
        """Serialize to JSON string or file"""
        json_str = json.dumps(self.to_dict(), indent=2)
        if filename:
            with open(filename, 'w') as f:
                f.write(json_str)
        return json_str
        
    @classmethod
    def from_json(cls, json_data: str = None, filename: str = None) -> 'Genome':
        """Deserialize from JSON string or file"""
        if filename:
            with open(filename, 'r') as f:
                json_data = f.read()
        
        data = json.loads(json_data)
        return cls.from_dict(data)
        
    def self_repair(self) -> bool:
        """Perform self-repair operations on the genome"""
        repaired = False
        
        for name, tree in self.trees.items():
            # Check for very deep trees and prune if necessary
            if tree.get_depth() > 15:
                # Replace with simpler random tree
                self.trees[name] = create_random_node(max_depth=6)
                repaired = True
                
            # Check for very large trees
            if len(tree.get_all_nodes()) > 100:
                self.trees[name] = create_random_node(max_depth=5)
                repaired = True
                
        return repaired
        
    def get_all_nodes(self) -> List[Tuple[str, ASTNode]]:
        """Get all nodes from all trees with their tree names"""
        all_nodes = []
        for tree_name, tree in self.trees.items():
            nodes = tree.get_all_nodes()
            for node in nodes:
                all_nodes.append((tree_name, node))
        return all_nodes
        
    def replace_subtree(self, tree_name: str, old_node: ASTNode, new_node: ASTNode) -> bool:
        """Replace a subtree in the specified tree"""
        def replace_in_node(current_node: ASTNode) -> ASTNode:
            if current_node is old_node:
                return new_node
            
            if hasattr(current_node, 'children'):
                if hasattr(current_node, 'child'):  # UnaryOp, NoiseOp
                    current_node.child = replace_in_node(current_node.child)
                    current_node.children = [current_node.child]
                elif hasattr(current_node, 'left') and hasattr(current_node, 'right'):  # BinaryOp
                    current_node.left = replace_in_node(current_node.left)
                    current_node.right = replace_in_node(current_node.right)
                    current_node.children = [current_node.left, current_node.right]
                    
            return current_node
            
        if tree_name in self.trees:
            if self.trees[tree_name] is old_node:
                self.trees[tree_name] = new_node
                return True
            else:
                self.trees[tree_name] = replace_in_node(self.trees[tree_name])
                return True
        return False
        
    def __str__(self) -> str:
        """String representation of the genome"""
        lines = [f"Genome ({self.mode} mode):"]
        lines.append(f"  Fitness: {self.fitness:.4f} (novelty: {self.novelty:.4f}, aesthetic: {self.aesthetic:.4f})")
        lines.append(f"  Complexity: {self.get_complexity()}, Depth: {self.get_depth()}, Age: {self.age}")
        
        for name, tree in self.trees.items():
            lines.append(f"  {name}: {str(tree)[:100]}{'...' if len(str(tree)) > 100 else ''}")
            
        return '\n'.join(lines)