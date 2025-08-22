"""
ast_evolution/cli.py - Command-line interface
"""
import click
import os
import time
from typing import Optional
import numpy as np

from .population import Population
from .fitness import FitnessEvaluator
from .evaluator import Evaluator
from .archive import EvolutionArchive
from .genome import Genome

@click.group()
def cli():
    """AST Evolution - Self-evolving generative algorithms"""
    pass

@cli.command()
@click.option('--generations', '-g', default=50, help='Number of generations to evolve')
@click.option('--population', '-p', default=40, help='Population size')
@click.option('--mode', '-m', type=click.Choice(['image', 'audio']), default='image', 
              help='Generation mode: image or audio')
@click.option('--out', '-o', default='out/', help='Output directory')
@click.option('--mutation-rate', default=0.1, help='Mutation rate (0.0-1.0)')
@click.option('--crossover-rate', default=0.7, help='Crossover rate (0.0-1.0)')
@click.option('--novelty-weight', default=0.7, help='Weight for novelty in fitness')
@click.option('--aesthetic-weight', default=0.3, help='Weight for aesthetics in fitness')
@click.option('--eval-size', default=64, help='Evaluation resolution (images) or duration (audio)')
@click.option('--elite-size', default=2, help='Number of elite individuals to preserve')
@click.option('--save-every', default=10, help='Save best genomes every N generations')
@click.option('--render-best', default=5, help='Render best N individuals each generation')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def evolve(generations, population, mode, out, mutation_rate, crossover_rate,
           novelty_weight, aesthetic_weight, eval_size, elite_size, save_every,
           render_best, verbose):
    """Evolve a population of generative algorithms"""
    
    # Setup
    os.makedirs(out, exist_ok=True)
    
    click.echo(f"Starting evolution: {generations} generations, population {population}")
    click.echo(f"Mode: {mode}, Output: {out}")
    
    # Initialize components
    pop = Population(population, mode)
    fitness_eval = FitnessEvaluator(novelty_weight, aesthetic_weight)
    evaluator = Evaluator()
    archive = EvolutionArchive(os.path.join(out, "archive"))
    
    # Load existing feature archive if available
    existing_features = archive.load_feature_archive()
    if existing_features:
        fitness_eval.novelty_search.archive = existing_features
        click.echo(f"Loaded {len(existing_features)} features from existing archive")
    
    start_time = time.time()
    
    # Evolution loop
    for gen in range(generations):
        gen_start_time = time.time()
        
        # Evaluate population
        if mode == 'image':
            evaluation_size = (eval_size, eval_size)
            eval_data = evaluator.batch_evaluate_population(pop, size=evaluation_size)
        else:
            eval_duration = min(eval_size, 5.0)  # Limit audio duration
            eval_data = evaluator.batch_evaluate_population(pop, duration=eval_duration)
        
        # Calculate fitness for each genome
        for i, genome in enumerate(pop.genomes):
            if i in eval_data:
                fitness, novelty, aesthetic = fitness_eval.evaluate_genome(genome, eval_data[i])
                genome.fitness = fitness
                genome.novelty = novelty
                genome.aesthetic = aesthetic
            else:
                genome.fitness = 0.0
                genome.novelty = 0.0
                genome.aesthetic = 0.0
        
        # Get statistics
        stats = pop.get_stats()
        
        # Archive generation
        if gen % save_every == 0 or gen == generations - 1:
            archive.archive_generation(pop, gen, stats)
            archive.save_feature_archive(fitness_eval.novelty_search.archive)
        
        # Render best individuals
        if gen % save_every == 0 or gen == generations - 1:
            best_genomes = pop.get_best(render_best)
            render_dir = os.path.join(out, "renders", f"gen_{gen:04d}")
            os.makedirs(render_dir, exist_ok=True)
            
            for i, genome in enumerate(best_genomes):
                if mode == 'image':
                    filename = os.path.join(render_dir, f"best_{i+1:02d}.png")
                    evaluator.render_image(genome, size=(512, 512), filename=filename)
                else:
                    filename = os.path.join(render_dir, f"best_{i+1:02d}.wav")
                    evaluator.render_audio(genome, duration=3.0, filename=filename)
        
        # Display progress
        gen_time = time.time() - gen_start_time
        total_time = time.time() - start_time
        
        if verbose or gen % 10 == 0 or gen == generations - 1:
            fitness_stats = stats.get('fitness', {})
            click.echo(f"Gen {gen:3d}/{generations}: "
                      f"Best={fitness_stats.get('max', 0):.4f} "
                      f"Avg={fitness_stats.get('mean', 0):.4f} "
                      f"Archive={fitness_eval.get_archive_size()} "
                      f"Time={gen_time:.1f}s")
            
            if verbose:
                complexity_stats = stats.get('complexity', {})
                click.echo(f"         Complexity: {complexity_stats.get('mean', 0):.1f}±{complexity_stats.get('std', 0):.1f} "
                          f"Depth: {stats.get('depth', {}).get('mean', 0):.1f}")
        
        # Evolve to next generation (except last)
        if gen < generations - 1:
            pop.evolve_generation(mutation_rate, crossover_rate, elite_size)
    
    # Final summary
    total_time = time.time() - start_time
    click.echo(f"\nEvolution completed in {total_time:.1f}s ({total_time/60:.1f} min)")
    
    # Generate summary report
    summary = archive.export_summary_report()
    click.echo("\nSummary report saved to archive/evolution_summary.txt")
    
    if verbose:
        click.echo("\nTOP 3 FINAL GENOMES:")
        best_final = pop.get_best(3)
        for i, genome in enumerate(best_final):
            click.echo(f"{i+1}. Fitness: {genome.fitness:.4f} "
                      f"(N: {genome.novelty:.3f}, A: {genome.aesthetic:.3f}) "
                      f"Complexity: {genome.get_complexity()}")

@cli.command()
@click.option('--genome', '-g', required=True, help='Path to genome JSON file')
@click.option('--mode', '-m', type=click.Choice(['image', 'audio']), help='Override mode')
@click.option('--t', default=0.0, help='Time parameter value')
@click.option('--size', default=512, help='Output size (image) or duration (audio)')
@click.option('--out', '-o', help='Output filename (optional)')
@click.option('--sample-rate', default=22050, help='Audio sample rate')
@click.option('--frames', default=0, help='Create animation with N frames')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def render(genome, mode, t, size, out, sample_rate, frames, verbose):
    """Render a genome from JSON file"""
    
    # Load genome
    try:
        g = Genome.from_json(filename=genome)
        click.echo(f"Loaded genome: {genome}")
        
        if verbose:
            click.echo(f"Mode: {g.mode}, Complexity: {g.get_complexity()}, Depth: {g.get_depth()}")
            click.echo(f"Fitness: {g.fitness:.4f} (Novelty: {g.novelty:.4f}, Aesthetic: {g.aesthetic:.4f})")
    
    except Exception as e:
        click.echo(f"Error loading genome: {e}")
        return
    
    # Override mode if specified
    if mode:
        g.mode = mode
    
    # Setup evaluator
    evaluator = Evaluator()
    
    # Generate output filename if not provided
    if not out:
        base_name = os.path.splitext(os.path.basename(genome))[0]
        if g.mode == 'image':
            if frames > 0:
                out = f"{base_name}_anim"
            else:
                out = f"{base_name}_t{t:.2f}.png"
        else:
            out = f"{base_name}_t{t:.2f}.wav"
    
    start_time = time.time()
    
    try:
        if g.mode == 'image':
            if frames > 0:
                # Create animation
                click.echo(f"Creating {frames} frame animation...")
                animation_frames = evaluator.create_animation_frames(
                    g, num_frames=frames, size=(size, size), 
                    t_range=(t, t + 2*np.pi)
                )
                
                # Save frames
                os.makedirs(out, exist_ok=True)
                for i, frame in enumerate(animation_frames):
                    frame_file = os.path.join(out, f"frame_{i:04d}.png")
                    frame.save(frame_file)
                
                click.echo(f"Animation frames saved to: {out}/")
                
            else:
                # Single image
                image = evaluator.render_image(g, size=(size, size), t=t, filename=out)
                click.echo(f"Image saved: {out}")
                
        else:  # audio
            audio = evaluator.render_audio(g, duration=size, sample_rate=sample_rate, 
                                         t_start=t, filename=out)
            click.echo(f"Audio saved: {out}")
            
        render_time = time.time() - start_time
        if verbose:
            click.echo(f"Render time: {render_time:.2f}s")
            
    except Exception as e:
        click.echo(f"Error during rendering: {e}")

@cli.command()
@click.option('--archive', '-a', default='out/archive', help='Archive directory path')
@click.option('--top', default=10, help='Show top N genomes')
def analyze(archive, top):
    """Analyze evolution results from archive"""
    
    try:
        arch = EvolutionArchive(archive)
        
        # Load evolution log
        import json
        log_file = os.path.join(archive, "logs", "evolution_log.json")
        if not os.path.exists(log_file):
            click.echo(f"No evolution log found at {log_file}")
            return
            
        with open(log_file, 'r') as f:
            arch.evolution_log = json.load(f)
        
        # Generate and display summary
        summary = arch.export_summary_report()
        click.echo(summary)
        
        # Archive statistics
        stats = arch.get_archive_stats()
        click.echo(f"\nArchive Statistics:")
        click.echo(f"Total files: {stats['disk_files']}")
        click.echo(f"Archive size: {stats['archive_size_mb']:.1f} MB")
        
        # List top genome files
        genome_files = arch.list_archived_genomes()
        if genome_files:
            click.echo(f"\nRecent genome files:")
            for file in sorted(genome_files)[-10:]:
                click.echo(f"  {file}")
                
    except Exception as e:
        click.echo(f"Error analyzing archive: {e}")

if __name__ == '__main__':
    cli()