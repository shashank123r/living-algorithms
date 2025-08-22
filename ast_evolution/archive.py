"""
ast_evolution/archive.py - Evolution archive and persistence
"""
import os
import json
import time
from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime

from .genome import Genome
from .population import Population

class EvolutionArchive:
    """Archive evolution runs, genomes, and feature vectors"""
    
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.evolution_log = []
        
        # Create directory structure
        self.dirs = {
            'genomes': os.path.join(base_path, 'genomes'),
            'features': os.path.join(base_path, 'features'),
            'logs': os.path.join(base_path, 'logs'),
            'renders': os.path.join(base_path, 'renders'),
            'stats': os.path.join(base_path, 'stats')
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
    
    def archive_generation(self, population: Population, generation: int, 
                          stats: Dict[str, Any]) -> None:
        """Archive a generation's data"""
        timestamp = time.time()
        
        # Save best genomes
        best_genomes = population.get_best(5)
        genome_files = []
        
        for i, genome in enumerate(best_genomes):
            filename = f"gen_{generation:04d}_rank_{i+1:02d}.json"
            filepath = os.path.join(self.dirs['genomes'], filename)
            genome.to_json(filepath)
            genome_files.append(filename)
        
        # Log generation data
        generation_data = {
            'generation': generation,
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp).isoformat(),
            'stats': stats,
            'best_genomes': genome_files,
            'population_diversity': population.diversity_stats()
        }
        
        self.evolution_log.append(generation_data)
        
        # Save evolution log
        log_file = os.path.join(self.dirs['logs'], 'evolution_log.json')
        with open(log_file, 'w') as f:
            json.dump(self.evolution_log, f, indent=2)
        
        # Save generation stats
        stats_file = os.path.join(self.dirs['stats'], f'gen_{generation:04d}_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(generation_data, f, indent=2)
    
    def save_feature_archive(self, feature_archive: List[np.ndarray]) -> None:
        """Save novelty search feature archive"""
        if feature_archive:
            archive_array = np.array(feature_archive)
            feature_file = os.path.join(self.dirs['features'], 'novelty_archive.npy')
            np.save(feature_file, archive_array)
            
            # Also save metadata
            metadata = {
                'num_features': len(feature_archive),
                'feature_dimension': archive_array.shape[1] if len(archive_array.shape) > 1 else 0,
                'timestamp': time.time(),
                'datetime': datetime.now().isoformat()
            }
            
            metadata_file = os.path.join(self.dirs['features'], 'novelty_archive_meta.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def load_feature_archive(self) -> Optional[List[np.ndarray]]:
        """Load existing feature archive"""
        feature_file = os.path.join(self.dirs['features'], 'novelty_archive.npy')
        
        try:
            if os.path.exists(feature_file):
                archive_array = np.load(feature_file)
                return [arr for arr in archive_array]
        except Exception as e:
            print(f"Could not load feature archive: {e}")
            
        return None
    
    def export_summary_report(self) -> str:
        """Generate and save a summary report of the evolution run"""
        if not self.evolution_log:
            return "No evolution data to summarize"
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("EVOLUTION SUMMARY REPORT")
        report_lines.append("=" * 60)
        
        # Basic info
        first_gen = self.evolution_log[0]
        last_gen = self.evolution_log[-1]
        
        report_lines.append(f"Generations: {len(self.evolution_log)}")
        report_lines.append(f"Start time: {first_gen['datetime']}")
        report_lines.append(f"End time: {last_gen['datetime']}")
        
        # Duration
        duration = last_gen['timestamp'] - first_gen['timestamp']
        report_lines.append(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        
        # Population info
        first_stats = first_gen['stats']
        last_stats = last_gen['stats']
        
        report_lines.append(f"Population size: {first_stats.get('population_size', 'N/A')}")
        report_lines.append(f"Mode: {first_gen.get('mode', 'Unknown')}")
        
        report_lines.append("\n" + "-" * 40)
        report_lines.append("FITNESS EVOLUTION")
        report_lines.append("-" * 40)
        
        # Fitness progression
        report_lines.append(f"Initial best fitness: {first_stats.get('fitness', {}).get('max', 0):.4f}")
        report_lines.append(f"Final best fitness: {last_stats.get('fitness', {}).get('max', 0):.4f}")
        report_lines.append(f"Initial avg fitness: {first_stats.get('fitness', {}).get('mean', 0):.4f}")
        report_lines.append(f"Final avg fitness: {last_stats.get('fitness', {}).get('mean', 0):.4f}")
        
        # Complexity evolution
        report_lines.append("\n" + "-" * 40)
        report_lines.append("COMPLEXITY EVOLUTION")
        report_lines.append("-" * 40)
        
        initial_complexity = first_stats.get('complexity', {})
        final_complexity = last_stats.get('complexity', {})
        
        report_lines.append(f"Initial avg complexity: {initial_complexity.get('mean', 0):.1f}")
        report_lines.append(f"Final avg complexity: {final_complexity.get('mean', 0):.1f}")
        report_lines.append(f"Initial max complexity: {initial_complexity.get('max', 0)}")
        report_lines.append(f"Final max complexity: {final_complexity.get('max', 0)}")
        
        # Diversity metrics
        report_lines.append("\n" + "-" * 40)
        report_lines.append("DIVERSITY METRICS")
        report_lines.append("-" * 40)
        
        initial_diversity = first_gen.get('population_diversity', {})
        final_diversity = last_gen.get('population_diversity', {})
        
        report_lines.append(f"Initial structural diversity: {initial_diversity.get('structural_diversity', 0):.3f}")
        report_lines.append(f"Final structural diversity: {final_diversity.get('structural_diversity', 0):.3f}")
        report_lines.append(f"Initial fitness diversity: {initial_diversity.get('fitness_diversity', 0):.3f}")
        report_lines.append(f"Final fitness diversity: {final_diversity.get('fitness_diversity', 0):.3f}")
        
        # Best performers over time
        report_lines.append("\n" + "-" * 40)
        report_lines.append("TOP PERFORMERS BY GENERATION")
        report_lines.append("-" * 40)
        
        # Sample every 10th generation or show all if few generations
        sample_gens = self.evolution_log[::max(1, len(self.evolution_log) // 10)]
        
        for gen_data in sample_gens[-10:]:  # Last 10 samples
            gen = gen_data['generation']
            fitness_stats = gen_data['stats'].get('fitness', {})
            best_fitness = fitness_stats.get('max', 0)
            avg_fitness = fitness_stats.get('mean', 0)
            
            report_lines.append(f"Gen {gen:3d}: Best={best_fitness:.4f}, Avg={avg_fitness:.4f}")
        
        # Archive statistics
        report_lines.append("\n" + "-" * 40)
        report_lines.append("ARCHIVE STATISTICS")
        report_lines.append("-" * 40)
        
        archive_stats = self.get_archive_stats()
        report_lines.append(f"Total archived genomes: {archive_stats.get('genome_files', 0)}")
        report_lines.append(f"Archive size: {archive_stats.get('archive_size_mb', 0):.1f} MB")
        report_lines.append(f"Feature archive size: {archive_stats.get('feature_archive_size', 0)}")
        
        # Best final genomes
        report_lines.append("\n" + "-" * 40)
        report_lines.append("FINAL BEST GENOMES")
        report_lines.append("-" * 40)
        
        best_files = last_gen.get('best_genomes', [])
        for i, filename in enumerate(best_files[:3]):
            genome_path = os.path.join(self.dirs['genomes'], filename)
            if os.path.exists(genome_path):
                try:
                    genome = Genome.from_json(filename=genome_path)
                    report_lines.append(f"#{i+1}: {filename}")
                    report_lines.append(f"     Fitness: {genome.fitness:.4f} "
                                       f"(N: {genome.novelty:.3f}, A: {genome.aesthetic:.3f})")
                    report_lines.append(f"     Complexity: {genome.get_complexity()}, "
                                       f"Depth: {genome.get_depth()}, Age: {genome.age}")
                except Exception as e:
                    report_lines.append(f"#{i+1}: {filename} (Error loading: {e})")
        
        report_text = "\n".join(report_lines)
        
        # Save report
        report_file = os.path.join(self.base_path, 'evolution_summary.txt')
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        return report_text
    
    def get_archive_stats(self) -> Dict[str, Any]:
        """Get archive storage statistics"""
        stats = {
            'genome_files': 0,
            'render_files': 0,
            'log_files': 0,
            'feature_archive_size': 0,
            'disk_files': 0,
            'archive_size_mb': 0.0
        }
        
        total_size = 0
        
        # Count files and calculate sizes
        for dir_name, dir_path in self.dirs.items():
            if os.path.exists(dir_path):
                for root, dirs, files in os.walk(dir_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if os.path.exists(file_path):
                            file_size = os.path.getsize(file_path)
                            total_size += file_size
                            stats['disk_files'] += 1
                            
                            if dir_name == 'genomes' and file.endswith('.json'):
                                stats['genome_files'] += 1
                            elif dir_name == 'renders':
                                stats['render_files'] += 1
                            elif dir_name == 'logs':
                                stats['log_files'] += 1
        
        stats['archive_size_mb'] = total_size / (1024 * 1024)
        
        # Feature archive specific stats
        feature_file = os.path.join(self.dirs['features'], 'novelty_archive.npy')
        if os.path.exists(feature_file):
            try:
                archive_array = np.load(feature_file)
                stats['feature_archive_size'] = len(archive_array)
            except:
                pass
        
        return stats
    
    def list_archived_genomes(self) -> List[str]:
        """List all archived genome files"""
        genome_files = []
        genomes_dir = self.dirs['genomes']
        
        if os.path.exists(genomes_dir):
            for file in os.listdir(genomes_dir):
                if file.endswith('.json'):
                    genome_files.append(os.path.join(genomes_dir, file))
        
        return sorted(genome_files)
    
    def load_genome(self, filename: str) -> Optional[Genome]:
        """Load a specific archived genome"""
        try:
            if not os.path.isabs(filename):
                filename = os.path.join(self.dirs['genomes'], filename)
                
            if os.path.exists(filename):
                return Genome.from_json(filename=filename)
        except Exception as e:
            print(f"Error loading genome {filename}: {e}")
            
        return None
    
    def cleanup_old_archives(self, keep_generations: int = 100) -> int:
        """Clean up old archive files, keeping only recent generations"""
        removed_count = 0
        
        # Get all genome files
        genomes_dir = self.dirs['genomes']
        if not os.path.exists(genomes_dir):
            return 0
        
        genome_files = []
        for file in os.listdir(genomes_dir):
            if file.startswith('gen_') and file.endswith('.json'):
                try:
                    # Extract generation number
                    gen_str = file.split('_')[1]
                    gen_num = int(gen_str)
                    genome_files.append((gen_num, file))
                except:
                    continue
        
        # Sort by generation
        genome_files.sort(reverse=True)  # Newest first
        
        # Remove old files
        for i, (gen_num, filename) in enumerate(genome_files):
            if i >= keep_generations * 5:  # Keep top 5 per generation
                file_path = os.path.join(genomes_dir, filename)
                try:
                    os.remove(file_path)
                    removed_count += 1
                except:
                    pass
        
        return removed_count
    
    def export_best_genomes(self, n: int = 10) -> List[Genome]:
        """Export the best n genomes across all generations"""
        best_genomes = []
        
        # Load all generation logs and find best performers
        fitness_records = []
        
        for gen_data in self.evolution_log:
            gen = gen_data['generation']
            best_files = gen_data.get('best_genomes', [])
            
            for filename in best_files:
                genome_path = os.path.join(self.dirs['genomes'], filename)
                genome = self.load_genome(genome_path)
                if genome:
                    fitness_records.append((genome.fitness, genome, gen, filename))
        
        # Sort by fitness and take top n
        fitness_records.sort(key=lambda x: x[0], reverse=True)
        
        for fitness, genome, gen, filename in fitness_records[:n]:
            best_genomes.append(genome)
        
        return best_genomes