# Grain Boundary Generator

This project provides tools to generate grain boundaries (GBs) for various crystal systems using `pymatgen`. It supports Cubic, Tetragonal, Orthorhombic, Rhombohedral, and Hexagonal systems.

## Features

- **Generate GBs from parameters**: Specify rotation axis, rotation angle (or sigma value), and GB plane.
- **Generate GBs from matrices**: Use specific transformation matrices for the grains.
- **Enumerate Sigma values**: Find possible Sigma values and rotation angles for a given rotation axis and crystal system.
- **Support for various systems**: Works with fcc, bcc, and other crystal structures.

## Dependencies

- Python 3.x
- `numpy`
- `pymatgen`
- `monty`

## Installation

Ensure you have the required packages installed:

```bash
pip install numpy pymatgen monty
```

## Usage

### Running Tests

1.  **Prepare Input Files**: Place your CIF files (e.g., `Cu_mp-30_primitive.cif`) in the `inputfile` directory located in the same folder as the script.
2.  **Run the Test Script**:

    ```bash
    python -m pytest main.py
    ```

### Using GBGenerator in Your Code

```python
from pymatgen.core import Structure
from grain_boundary_generator import GBGenerator

# Load your structure
initial_struct = Structure.from_file("path/to/your/structure.cif")

# Initialize the generator
gb_gen = GBGenerator(initial_struct)

# Generate a grain boundary
# Example: Sigma 5 twist boundary for FCC Cu with [001] rotation axis
gb = gb_gen.gb_from_parameters(
    rotation_axis=[0, 0, 1],
    rotation_angle=36.87,  # Approximately 36.87 degrees for Sigma 5
    plane=[0, 0, 1],
    expand_times=2
)

# Output the result
gb.to(filename="GB_Sigma5.cif")
```

## Directory Structure

- `grain_boundary_generator.py`: Core logic for GB generation.
- `main.py`: Test script to verify functionality.
- `inputfile/`: Directory to store input CIF files for testing.

## Author

Xiang-Guo Li
