# codigoTFG

Scripts used for the experiments of the thesis. They must be run from this folder, because they read their configuration from a local `.txt` file. (Resumen: scripts de lanzamiento, configuraciones y post-procesado de frentes de Pareto del TFG.)

| File | Purpose |
| --- | --- |
| `antenaparche_NSGAII.py` | Launches NSGA-II on the patch-antenna problem (reads `conf_NSGAII.txt`) |
| `antenaparche_SMPSO.py` | Launches SMPSO on the same problem (reads `conf_SMPSO.txt`) |
| `conf_NSGAII.txt`, `conf_SMPSO.txt` | Multi-frequency scenario configuration (algorithm, operators, output folders, warm start, CST paths) |
| `conf_NSGAII_monofreq.txt`, `conf_SMPSO_monofreq.txt` | Single-frequency (5 GHz) scenario configuration |
| `Pareto_front.py` | Post-processing: loads the saved `FUN` files and plots the Pareto fronts |

## Before running

1. Install the package from the repository root: `pip install -e .`
2. Install CST Studio Suite and check the path of its Python libraries at the top of `src/jmetal/problem/multiobjective/antenaparche.py`.
3. Edit the paths in the configuration file you use. The provided files contain the absolute Windows paths of the original machine (`C:\Users\...`); replace them with your own folders. Do not put quotes around the paths.
4. Provide the CST macro template referenced by `ruta_plantilla_macro` (not included).
5. Set `usar_rescate = False` in the `[WARM_START]` section to start from a random population, or `True` plus the `VAR`/`FUN` files to resume a previous run.

```
cd codigoTFG
python antenaparche_SMPSO.py
```
