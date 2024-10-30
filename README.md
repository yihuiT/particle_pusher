## Required Python Files

- `constants.py`
- `particle.py`

## Required Python Libraries

- `numpy`
- `matplotlib`
- `multiprocessing`
- `h5py`

## Execution Command

To execute the simulation script, run:

`python main.py`

## Input Parameters

The `main.py` script takes the following input parameters:

- **`t_val`** (float): The simulation time.
- **`props`** (str): Specifies the particle type, either `"electron"` or `"proton"`.
- **`pusher`** (str): The particle pusher method to be used. Options include:
  - `"boris"`
  - `"hc"`
  - `"rk4"`
  - `"rk8"`
  - `"gca"`
- **`interp_method`** (str): The interpolation method to be used. Options include:
  - `'ana'` for analytical dipole field
  - `'tri'` for trilinear interpolation method
  - `'tsc'` for Triangular-Shaped-Cloud
  - `'bsp'` for B-spline
- **`dx`** (float): The grid resolution (in Earth radii).
- **`x_min`** (float): The minimum field boundary (in Earth radii).
- **`x_max`** (float): The maximum field boundary (in Earth radii).

## Output Data

The output data is formatted as HDF5 files. Example output files include:

- **`pr_pusher.h5`**: Proton data for each integrator (Figure 4).
- **`el_pusher.h5`**: Electron data for each integrator (Figure 5).
- **`el_grid.h5`**: Electron data for each interpolation method (Figure 6).

These files are provided in case difficulties are encountered during the simulation.

## Plotting the Results

Use the `plot.ipynb` Jupyter notebook to plot figures from the output data for the report.
