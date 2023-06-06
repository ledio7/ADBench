import os
import sys
import pathlib

path = pathlib.Path(__file__).parent.parent.resolve()
print(f"Path: {path}")

final_path = path.joinpath(os.path.join('Synthetic', 'Data_gen'))
print(f"Final path: {final_path}")

print(f"Dove itera: {os.path.join(final_path, 'Nome.csv')}")

path = pathlib.Path(__file__).parent.parent.resolve()
csv_file = path.joinpath(os.path.join('Synthetic', 'Data_gen'))
print(f"Dove controla se già esiste il dataset {os.path.join(csv_file, 'Nome.csv')}")

path = pathlib.Path(__file__).parent.parent.resolve()
path_agg = path.joinpath(os.path.join('Synthetic', 'Benchmark_agg.csv'))
print(f"Dove salva i csv finali: {path_agg}")
