# Study A analysis tools (archived 2026-10-30)

`run_3d_analysis_from_data.py`, `generalization_tests.py`, `cost_analysis.py` and `phase_space_analysis.py` are the Study A-era analysis scripts
(random-forest hurdle models, Jacobi-drift labels, FTLE as a feature). No Study B-D code imports them. They were moved here unchanged by
`git mv`, so their history is intact.

To rerun them exactly as in Study A, check out the Study A state (the sealed snapshot `CODE_SNAPSHOT.json` / `FREEZE.json` lists the hashes)
and run them from the repository root. From this folder they still run with the repository root on the path:

    PYTHONPATH=../.. python run_3d_analysis_from_data.py --help

The library functions they call (`orbit_ml_cr3bp.modeling_3d`, `search_3d`, ...) stay in the package because Study B code uses them too.
`validate_on_capstone.py` stays in the root: `run_rigor_suite_3d.py fidelity` imports it for the CAPSTONE ephemeris check.
