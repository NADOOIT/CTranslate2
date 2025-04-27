# Agent Plan: Make Fork a Drop-in Replacement for Official CTranslate2 (pip install)

## Objective
Ensure that this fork of CTranslate2 can be installed via pip (from Git or local path) and behaves as a true drop-in replacement for the official package, including building the Python native extension and matching the official Python API and directory layout.

---

## Step-by-Step Plan

### 1. **Audit and Sync Python Packaging**
- [ ] Compare `python/` directory structure and contents with the official CTranslate2 repo.
- [ ] Ensure all packaging files (`setup.py`, `pyproject.toml`, `CMakeLists.txt`) match upstream or are compatible.
- [ ] Confirm all required `.cc` files and Python modules are present.

### 2. **Test Clean pip Install**
- [ ] In a fresh virtual environment, run:
  ```sh
  pip install git+https://github.com/NADOOIT/CTranslate2.git
  python -c "import ctranslate2; print(ctranslate2.__version__)"
  ```
- [ ] Confirm that `ctranslate2` imports and the native extension (`_ext.*.so`) is present in `site-packages/ctranslate2`.

### 3. **Fix Any Build or Packaging Issues**
- [ ] If the build fails or no `.so` is installed, check:
    - CMake errors or missing dependencies
    - Extension sources and install paths
    - `setup.py`/`pyproject.toml` configuration
- [ ] Update packaging files or CMake as needed for compatibility.

### 4. **Automate and Document**
- [ ] Add or update a `README.md` in the fork with clear install/build instructions.
- [ ] Optionally, add a GitHub Actions workflow to test pip install and import on push/PR.

### 5. **(Optional) Build and Upload Wheels**
- [ ] Build a wheel (`pip wheel .` or `python -m build`) for your fork.
- [ ] Upload to a private index or GitHub Release for even easier installs (optional).

---

## Deliverables
- This plan and all supporting scripts/docs in the `Agent/` folder.
- A fork that can be installed via pip as a drop-in replacement for the official CTranslate2.
- Documentation and (optionally) CI for ongoing compatibility.

---

## Progress Tracking
- [ ] Audit complete
- [ ] Clean pip install tested
- [ ] Issues fixed
- [ ] Documentation updated
- [ ] (Optional) Wheel built and/or CI added
