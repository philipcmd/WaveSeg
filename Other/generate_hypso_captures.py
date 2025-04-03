import os
import hypso
from hypso import Hypso1
from hypso import Hypso2
from hypso.load import load_l1a_nc_cube
from hypso.load import load_l1b_nc_cube
from hypso.load import load_l1c_nc_cube
from hypso.load import load_l1d_nc_cube
from hypso.write import write_l1b_nc_file
from hypso.write import write_l1c_nc_file
from hypso.write import write_l1d_nc_file
from hypso.spectral_analysis import get_closest_wavelength_index





raw_data_path = r'C:\Users\Philip Shahdadfar\Downloads\trondheimdata-20250328T141220Z-001\trondheimdata'

# List of L1A files

raw_files = [
    'trondheim_2024-11-10T10-05-08Z-l1a.nc',
    'trondheim_2024-09-15T10-49-58Z-l1a.nc',
    'trondheim_2024-09-14T09-42-48Z-l1a.nc',
    'trondheim_2024-09-11T09-27-22Z-l1a.nc',
    'trondheim_2024-05-24T09-50-09Z-l1a.nc',
    'trondheim_2024-05-23T19-26-46Z-l1a.nc',
    'trondheim_2024-04-29T09-44-07Z-l1a.nc',
    'trondheim_2023-06-13T19-35-25Z-l1a.nc',
    'trondheim_2023-04-17T10-47-11Z-l1a.nc',
    'trondheim_2023-03-27T09-39-46Z-l1a.nc',
    'trondheim_2023-02-01T09-39-52Z-l1a.nc',
    'trondheim_2022-08-23T10-26-45Z-l1a.nc'
]



# Process each L1A file
for filename in raw_files:
    try:
        # Full path for input L1A file
        l1a_file_path = os.path.join(raw_data_path, filename)
        # Process the L1A file exactly like your working example
        satobj_h1 = Hypso1(path=l1a_file_path, verbose=True)
        satobj_h1.generate_l1b_cube()
        satobj_h1.generate_l1c_cube()
        satobj_h1.generate_l1d_cube()
        # Write the L1B file (same way as in your example)
        write_l1d_nc_file(satobj=satobj_h1, overwrite=True)
        print(f":white_tick: Successfully created L1B file for: {filename}")
    except Exception as e:
        print(f":x: Error processing {filename}: {e}")



