import os
import numpy as np
from pymatgen.core import Structure, Lattice
from grain_boundary_generator import GBGenerator

def get_cu_structure():
    """
    Load Cu structure from file or create it if file is missing.
    """
    cif_path = os.path.join("inputfile", "Cu_mp-30_primitive.cif")
    if os.path.exists(cif_path):
        print(f"Loading Cu structure from {cif_path}")
        return Structure.from_file(cif_path)
    else:
        print("CIF file not found, creating FCC Cu structure manually.")
        # FCC Cu, a = 3.615 Angstrom
        # Primitive cell
        a = 3.615
        lattice = Lattice.from_parameters(a, a, a, 60, 60, 60)
        return Structure(lattice, ["Cu"], [[0, 0, 0]])

def generate_gb(sigma, axis, plane, angle, label, expand_times=2, vacuum=0.0, shift=None, rm_tol=None):
    """
    Generate and save a grain boundary structure.
    """
    print(f"\nGenerating {label} GB...")
    print(f"  Sigma: {sigma}")
    print(f"  Axis: {axis}")
    print(f"  Plane: {plane}")
    print(f"  Angle: {angle:.2f}")
    if shift:
        print(f"  Shift: {shift}")
    if rm_tol:
        print(f"  Merge Tolerance: {rm_tol}")

    cu_struct = get_cu_structure()
    gb_gen = GBGenerator(cu_struct)
    
    gb = gb_gen.gb_from_parameters(
        rotation_axis=axis,
        rotation_angle=angle,
        plane=plane,
        expand_times=expand_times,
        vacuum_thickness=vacuum,
        shift=shift,
        rm_tol=rm_tol
    )
    
    # Construct filename
    # Format: Cu_Sigma{sigma}_Axis{axis}_Plane{plane}_Angle{angle}.cif
    axis_str = "".join(map(str, axis))
    plane_str = "".join(map(str, plane))
    filename = f"Cu_Sigma{sigma}_Axis{axis_str}_Plane{plane_str}_Angle{angle:.2f}"
    
    if shift:
        shift_str = "_Shift_" + "_".join([f"{s:.2f}" for s in shift])
        filename += shift_str
    if rm_tol:
        filename += f"_Merge{rm_tol}"
        
    filename += ".cif"
    
    # Save to file
    gb.to(filename=filename)
    print(f"  Saved to {filename}")

def main():
    # 1. Sigma 5 [001] Twist Boundary
    # Axis: [001], Angle: 36.87 (approx), Plane: [001] (Twist)
    # Exact angle for Sigma 5 is ~36.86989765
    sigma5_angle = GBGenerator.get_rotation_angle_from_sigma(5, [0, 0, 1])[0]
    generate_gb(
        sigma=5,
        axis=[0, 0, 1],
        plane=[0, 0, 1],
        angle=sigma5_angle,
        label="Sigma 5 [001] Twist"
    )

    # 2. Sigma 5 [001] Tilt Boundary (Example from image)
    # Image mentions Sigma 5 [100](012) - wait, [100] axis, (012) plane.
    # Let's try to reproduce something similar to the image description:
    # "Fe Sigma 5 [100](012) GB" -> Rotation axis [100], Plane (012)
    # For Cu (FCC), let's try a common Tilt GB.
    # Sigma 5 [001] (210) or (310) are common symmetric tilt boundaries.
    
    # Let's verify rotation angle for Sigma 5 with [001] axis
    # It returns multiple angles, usually 36.87 and 53.13
    
    # Generate Sigma 5 [001] (310) Symmetric Tilt
    generate_gb(
        sigma=5,
        axis=[0, 0, 1],
        plane=[3, 1, 0],
        angle=sigma5_angle,
        label="Sigma 5 [001] (310) Tilt"
    )

    # 3. Demonstrate "Shift" feature
    # Apply a small shift to the top grain
    generate_gb(
        sigma=5,
        axis=[0, 0, 1],
        plane=[0, 0, 1],
        angle=sigma5_angle,
        label="Sigma 5 [001] Twist with Shift",
        shift=[0.1, 0.1, 0.0] # Shift in a, b directions
    )

    # 4. Demonstrate "Merge" feature
    # Use a large tolerance to force merging of close atoms (e.g. at interface)
    generate_gb(
        sigma=5,
        axis=[0, 0, 1],
        plane=[0, 0, 1],
        angle=sigma5_angle,
        label="Sigma 5 [001] Twist with Merge",
        rm_tol=0.5 # 0.5 Angstrom tolerance
    )

    print("\nAll GBs generated successfully.")

if __name__ == "__main__":
    main()
