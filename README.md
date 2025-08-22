# AST Evolution - Self-Evolving Generative Algorithms

A genetic programming system that evolves **expression trees (ASTs)** to generate images and audio. Each individual in the population is represented as a typed expression tree that maps input coordinates `(x, y, t)` to RGB values or audio samples.

## Features

- **AST-based genomes**: Expression trees with safe mathematical primitives
- **Dual-mode generation**: Images (RGB) and audio synthesis
- **Novelty search**: Feature-based diversity preservation with archive
- **Aesthetic evaluation**: Simple proxies for visual/audio quality
- **Genetic operators**: Mutation (point, subtree, shrink, grow) and crossover
- **Self-repair**: Automatic genome validation and fixing
- **JSON serialization**: Save/load evolved algorithms
- **CLI interface**: Easy evolution and rendering commands

## Installation

```bash
# Clone or create the project directory
mkdir ast_evolution_project
cd ast_evolution_project

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Project Structure

```
ast_evolution/
├── __init__.py          # Package initialization
├── ast_nodes.py         # AST node types and primitives
├── genome.py           # Genome representation and serialization  
├── population.py       # Population management and genetic operators
├── fitness.py          # Novelty search and aesthetic evaluation
├── evaluator.py        # Genome evaluation and rendering
├── archive.py          # Evolution archive and persistence
└── cli.py              # Command-line interface

tests/
└── test_genome.py      # Unit tests

requirements.txt        # Python dependencies
README.md              # This file
```

## Quick Start

### 1. Evolve Image Generators

```bash
# Basic evolution run
python -m ast_evolution.cli evolve --generations 50 --population 40 --mode image --out results/

# With custom parameters
python -m ast_evolution.cli evolve \
  --generations 100 \
  --population 60 \
  --mode image \
  --mutation-rate 0.15 \
  --crossover-rate 0.8 \
  --novelty-weight 0.6 \
  --aesthetic-weight 0.4 \
  --out results/images/
```

### 2. Evolve Audio Generators

```bash
# Audio evolution
python -m ast_evolution.cli evolve --mode audio --generations 30 --population 30 --out results/audio/
```

### 3. Render Genomes

```bash
# Render single image
python -m ast_evolution.cli render \
  --genome results/genomes/gen_0049_rank_01.json \
  --mode image \
  --size 1024 \
  --t 3.14 \
  --out masterpiece.png

# Create animation (multiple frames with varying t)
python -m ast_evolution.cli render \
  --genome results/genomes/gen_0049_rank_01.json \
  --frames 60 \
  --out animation_frames/

# Render audio
python -m ast_evolution.cli render \
  --genome results/genomes/gen_0029_rank_01.json \
  --mode audio \
  --size 5.0 \
  --out generated_audio.wav
```

### 4. Analyze Results

```bash
# View evolution summary
python -m ast_evolution.cli analyze --archive results/archive/ --top 5
```

## Algorithm Details

### AST Representation

Each genome contains expression trees mapping `(x, y, t)` coordinates to outputs:

- **Image mode**: 3 trees for RGB channels
- **Audio mode**: 1 tree for audio samples

**Available primitives**:
- Variables: `x, y, t`  
- Constants: Random floats in [-2, 2]
- Unary ops: `sin, cos, tanh, abs, sqrt, neg`
- Binary ops: `add, sub, mul, div, mod, pow` (with safe guards)
- Noise: Perlin-like noise function

### Genetic Operators

**Mutation types**:
- Point mutation: Change operators or constants
- Subtree mutation: Replace entire subtrees  
- Shrink mutation: Replace complex nodes with simpler ones
- Grow mutation: Add complexity to terminal nodes

**Crossover**: Swap compatible subtrees between parents

**Self-repair**: Automatically fix overly deep/complex trees

### Fitness Evaluation

**Novelty Search** (70% weight):
- Extract features from generated content
- Calculate distance to k-nearest neighbors in archive
- Archive novel individuals for diversity preservation

**Aesthetic Proxy** (30% weight):
- Images: Color variance, edge density, harmonic balance
- Audio: Dynamic range, spectral spread, harmonic content

## Example Evolution Run

```bash
$ python -m ast_evolution.cli evolve --generations 50 --population 40 --mode image --out results/ --verbose

Starting evolution: 50 generations, population 40
Mode: image, Output: results/

Gen   0/50: Best=0.3421 Avg=0.1834 Archive=15 Time=2.3s
Gen  10/50: Best=0.5672 Avg=0.3891 Archive=127 Time=2.1s
Gen  20/50: Best=0.7234 Avg=0.4567 Archive=203 Time=2.0s
Gen  30/50: Best=0.8123 Avg=0.5234 Archive=256 Time=1.9s
Gen  40/50: Best=0.8756 Avg=0.6012 Archive=287 Time=1.8s
Gen  49/50: Best=0.9234 Avg=0.6789 Archive=312 Time=1.7s

Evolution completed in 98.2s (1.6 min)

TOP 3 FINAL GENOMES:
1. Fitness: 0.9234 (N: 0.651, A: 0.724) Complexity: 47
2. Fitness: 0.8967 (N: 0.623, A: 0.701) Complexity: 52  
3. Fitness: 0.8834 (N: 0.612, A: 0.689) Complexity: 41
```

## Generated Files

**During evolution**:
- `results/genomes/gen_XXXX_rank_XX.json` - Best genomes each generation
- `results/archive/features/novelty_archive.npy` - Feature archive for novelty search
- `results/archive/logs/evolution_log.json` - Detailed evolution log
- `results/renders/gen_XXXX/best_XX.png` - Rendered outputs

**After completion**:
- `results/evolution_summary.txt` - Human-readable summary report

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Tests cover:
- Genome creation, mutation, crossover validity
- AST node evaluation and serialization
- Population genetic operators
- Safe numeric operations

## Advanced Usage

### Custom Fitness Weights

```bash
# Prioritize novelty over aesthetics
python -m ast_evolution.cli evolve --novelty-weight 0.9 --aesthetic-weight 0.1 --mode image
```

### Population Parameters

```bash
# Large population, high mutation
python -m ast_evolution.cli evolve \
  --population 100 \
  --mutation-rate 0.2 \
  --crossover-rate 0.6 \
  --elite-size 5
```

### Batch Rendering

```bash
# Render multiple genomes from a generation
for genome in results/genomes/gen_0049_rank_*.json; do
  python -m ast_evolution.cli render --genome "$genome" --size 512 --out "gallery/$(basename "$genome" .json).png"
done
```

## Performance Notes

- Uses NumPy vectorization for fast evaluation
- Optional Numba JIT compilation for hotspots
- Batch evaluation of populations
- Archive size limits prevent memory growth

## Extending the System

**Add new primitives** in `ast_nodes.py`:
```python
class MyCustomOp(UnaryOp):
    def evaluate(self, x, y, t):
        child_val = self.child.evaluate(x, y, t)
        return my_custom_function(child_val)
```

**Custom fitness functions** in `fitness.py`:
```python
def my_aesthetic_metric(data):
    # Your custom aesthetic evaluation
    return score
```

## Requirements

- Python 3.8+
- NumPy, SciPy, Pillow
- librosa (audio processing)
- Click (CLI)
- pytest (testing)

## License

Open source - feel free to modify and extend!