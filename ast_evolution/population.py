"""
ast_evolution/population.py - Population management and genetic operators
"""
import random
import copy
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from .genome import Genome
from .ast_nodes import (ASTNode, Variable, Constant, UnaryOp, BinaryOp, NoiseOp, 
                       create_random_node, VARIABLES, UNARY_OPS, BINARY_OPS)

class Population:
    """Manages a population of genomes with genetic operators"""
    
    def __init__(self, size: int, mode: str = 'image'):
        self.size = size
        self.mode = mode
        self.genomes = []
        self.generation = 0
        
        # Initialize population
        for _ in range(size):
            genome = Genome(mode=mode)
            self.genomes.append(genome)
            
    def mutate(self, genome: Genome, mutation_rate: float = 0.1) -> Genome:
        """Apply mutation operations to a genome"""
        mutant = genome.copy()
        
        # Apply multiple mutation types with different probabilities
        if random.random() < mutation_rate:
            mutant = self._point_mutation(mutant)
            
        if random.random() < mutation_rate * 0.5:
            mutant = self._subtree_mutation(mutant)
            
        if random.random() < mutation_rate * 0.3:
            mutant = self._shrink_mutation(mutant)
            
        if random.random() < mutation_rate * 0.2:
            mutant = self._grow_mutation(mutant)
            
        # Self-repair after mutation
        mutant.self_repair()
        mutant.age = 0  # Reset age after mutation
        
        return mutant
        
    def _point_mutation(self, genome: Genome) -> Genome:
        """Point mutation: change operators or constants"""
        all_nodes = genome.get_all_nodes()
        if not all_nodes:
            return genome
            
        tree_name, node = random.choice(all_nodes)
        
        # Create new node based on type
        new_node = None
        
        if isinstance(node, Constant):
            # Mutate constant value
            new_value = node.value + random.gauss(0, 0.5)
            new_node = Constant(np.clip(new_value, -10, 10))
            
        elif isinstance(node, UnaryOp):
            # Change unary operator
            new_op = random.choice(UNARY_OPS)
            new_node = UnaryOp(new_op, node.child.copy())
            
        elif isinstance(node, BinaryOp):
            # Change binary operator
            new_op = random.choice(BINARY_OPS)
            new_node = BinaryOp(new_op, node.left.copy(), node.right.copy())
            
        elif isinstance(node, Variable):
            # Change variable
            new_var = random.choice(VARIABLES)
            new_node = Variable(new_var)
            
        elif isinstance(node, NoiseOp):
            # Change noise seed
            new_node = NoiseOp(node.child.copy(), random.randint(0, 1000000))
            
        if new_node:
            genome.replace_subtree(tree_name, node, new_node)
            
        return genome
        
    def _subtree_mutation(self, genome: Genome) -> Genome:
        """Subtree mutation: replace entire subtree"""
        all_nodes = genome.get_all_nodes()
        if not all_nodes:
            return genome
            
        tree_name, node = random.choice(all_nodes)
        
        # Create random replacement subtree
        max_depth = max(1, 6 - node.get_depth())
        new_subtree = create_random_node(max_depth=max_depth)
        
        genome.replace_subtree(tree_name, node, new_subtree)
        return genome
        
    def _shrink_mutation(self, genome: Genome) -> Genome:
        """Shrink mutation: replace subtree with smaller one"""
        all_nodes = genome.get_all_nodes()
        if not all_nodes:
            return genome
            
        # Find nodes with children
        complex_nodes = [(name, node) for name, node in all_nodes 
                        if hasattr(node, 'children') and len(node.get_all_nodes()) > 3]
        
        if complex_nodes:
            tree_name, node = random.choice(complex_nodes)
            
            # Replace with one of its children or a simple node
            if hasattr(node, 'child'):  # UnaryOp, NoiseOp
                replacement = node.child.copy()
            elif hasattr(node, 'left') and hasattr(node, 'right'):  # BinaryOp
                replacement = random.choice([node.left, node.right]).copy()
            else:
                replacement = create_random_node(max_depth=2)
                
            genome.replace_subtree(tree_name, node, replacement)
            
        return genome
        
    def _grow_mutation(self, genome: Genome) -> Genome:
        """Grow mutation: add complexity to tree"""
        all_nodes = genome.get_all_nodes()
        if not all_nodes:
            return genome
            
        # Find terminal nodes to expand
        terminal_nodes = [(name, node) for name, node in all_nodes 
                         if isinstance(node, (Variable, Constant))]
        
        if terminal_nodes:
            tree_name, node = random.choice(terminal_nodes)
            
            # Wrap in a new operation
            if random.random() < 0.5:
                # Unary operation
                op = random.choice(UNARY_OPS)
                new_node = UnaryOp(op, node.copy())
            else:
                # Binary operation
                op = random.choice(BINARY_OPS)
                other_child = create_random_node(max_depth=3)
                if random.random() < 0.5:
                    new_node = BinaryOp(op, node.copy(), other_child)
                else:
                    new_node = BinaryOp(op, other_child, node.copy())
                    
            genome.replace_subtree(tree_name, node, new_node)
            
        return genome
        
    def crossover(self, parent1: Genome, parent2: Genome) -> Tuple[Genome, Genome]:
        """Crossover: exchange subtrees between parents"""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Get all nodes from both parents
        nodes1 = child1.get_all_nodes()
        nodes2 = child2.get_all_nodes()
        
        if nodes1 and nodes2:
            # Select random nodes to swap
            tree_name1, node1 = random.choice(nodes1)
            tree_name2, node2 = random.choice(nodes2)
            
            # Only swap if trees are compatible (same mode implications)
            if self._compatible_for_swap(tree_name1, tree_name2, child1.mode):
                # Create copies of subtrees
                subtree1 = node1.copy()
                subtree2 = node2.copy()
                
                # Swap subtrees
                child1.replace_subtree(tree_name1, node1, subtree2)
                child2.replace_subtree(tree_name2, node2, subtree1)
        
        # Reset ages
        child1.age = 0
        child2.age = 0
        
        return child1, child2
    
    def _compatible_for_swap(self, tree_name1: str, tree_name2: str, mode: str) -> bool:
        """Check if tree names are compatible for crossover"""
        if mode == 'image':
            # RGB channels are interchangeable
            return tree_name1 in ['r', 'g', 'b'] and tree_name2 in ['r', 'g', 'b']
        else:  # audio
            # Only audio with audio
            return tree_name1 == 'audio' and tree_name2 == 'audio'
    
    def tournament_selection(self, tournament_size: int = 3) -> Genome:
        """Tournament selection"""
        tournament = random.sample(self.genomes, min(tournament_size, len(self.genomes)))
        return max(tournament, key=lambda g: g.fitness)
    
    def roulette_selection(self) -> Genome:
        """Roulette wheel selection"""
        # Ensure all fitness values are non-negative
        min_fitness = min(g.fitness for g in self.genomes)
        adjusted_fitness = [g.fitness - min_fitness + 0.01 for g in self.genomes]
        
        total_fitness = sum(adjusted_fitness)
        if total_fitness <= 0:
            return random.choice(self.genomes)
            
        pick = random.uniform(0, total_fitness)
        current = 0
        
        for i, fitness in enumerate(adjusted_fitness):
            current += fitness
            if current >= pick:
                return self.genomes[i]
                
        return self.genomes[-1]  # Fallback
    
    def evolve_generation(self, mutation_rate: float = 0.1, 
                         crossover_rate: float = 0.7, 
                         elite_size: int = 2) -> None:
        """Evolve to the next generation"""
        # Sort by fitness
        self.genomes.sort(key=lambda g: g.fitness, reverse=True)
        
        new_genomes = []
        
        # Elite preservation
        for i in range(min(elite_size, len(self.genomes))):
            elite = self.genomes[i].copy()
            elite.age += 1
            new_genomes.append(elite)
        
        # Generate offspring
        while len(new_genomes) < self.size:
            if random.random() < crossover_rate:
                # Crossover
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                child1, child2 = self.crossover(parent1, parent2)
                
                # Mutate children
                child1 = self.mutate(child1, mutation_rate)
                child2 = self.mutate(child2, mutation_rate)
                
                new_genomes.append(child1)
                if len(new_genomes) < self.size:
                    new_genomes.append(child2)
            else:
                # Mutation only
                parent = self.tournament_selection()
                child = self.mutate(parent, mutation_rate)
                new_genomes.append(child)
        
        # Update population
        self.genomes = new_genomes[:self.size]
        self.generation += 1
        
        # Incremental aging
        for genome in self.genomes:
            if hasattr(genome, 'age'):
                genome.age += 1
    
    def get_best(self, n: int = 1) -> List[Genome]:
        """Get the best n genomes"""
        sorted_genomes = sorted(self.genomes, key=lambda g: g.fitness, reverse=True)
        return sorted_genomes[:n]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get population statistics"""
        if not self.genomes:
            return {}
            
        fitnesses = [g.fitness for g in self.genomes]
        novelties = [g.novelty for g in self.genomes]
        aesthetics = [g.aesthetic for g in self.genomes]
        complexities = [g.get_complexity() for g in self.genomes]
        depths = [g.get_depth() for g in self.genomes]
        ages = [getattr(g, 'age', 0) for g in self.genomes]
        
        return {
            'generation': self.generation,
            'population_size': len(self.genomes),
            'fitness': {
                'min': min(fitnesses),
                'max': max(fitnesses),
                'mean': np.mean(fitnesses),
                'std': np.std(fitnesses)
            },
            'novelty': {
                'min': min(novelties),
                'max': max(novelties),
                'mean': np.mean(novelties),
                'std': np.std(novelties)
            },
            'aesthetic': {
                'min': min(aesthetics),
                'max': max(aesthetics),
                'mean': np.mean(aesthetics),
                'std': np.std(aesthetics)
            },
            'complexity': {
                'min': min(complexities),
                'max': max(complexities),
                'mean': np.mean(complexities),
                'std': np.std(complexities)
            },
            'depth': {
                'min': min(depths),
                'max': max(depths),
                'mean': np.mean(depths),
                'std': np.std(depths)
            },
            'age': {
                'min': min(ages),
                'max': max(ages),
                'mean': np.mean(ages),
                'std': np.std(ages)
            }
        }
    
    def diversity_stats(self) -> Dict[str, float]:
        """Calculate population diversity metrics"""
        if len(self.genomes) < 2:
            return {'structural_diversity': 0.0, 'fitness_diversity': 0.0}
            
        # Structural diversity - based on tree structures
        structures = []
        for genome in self.genomes:
            structure_sig = []
            for tree_name, tree in genome.trees.items():
                nodes = tree.get_all_nodes()
                node_types = [type(node).__name__ for node in nodes]
                structure_sig.append('_'.join(sorted(node_types)))
            structures.append('|'.join(structure_sig))
        
        unique_structures = len(set(structures))
        structural_diversity = unique_structures / len(structures)
        
        # Fitness diversity
        fitnesses = [g.fitness for g in self.genomes]
        fitness_std = np.std(fitnesses)
        fitness_mean = np.mean(fitnesses)
        fitness_diversity = fitness_std / (abs(fitness_mean) + 1e-10)
        
        return {
            'structural_diversity': structural_diversity,
            'fitness_diversity': fitness_diversity,
            'unique_structures': unique_structures
        }