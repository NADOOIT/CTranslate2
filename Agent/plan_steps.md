# Agent Execution Steps: Drop-in CTranslate2 Fork

This file tracks the concrete steps and findings as we execute the plan from README.md.

---

## 1. Audit and Sync Python Packaging

### a. Directory Structure
- [x] `python/` exists with `setup.py`, `pyproject.toml`, and `cpp/` sources.
- [x] `python/ctranslate2/` exists with `__init__.py`, `_ext.*.so`, and all required submodules.

### b. Packaging Files
- [x] `setup.py` present, custom build logic for C++/Metal extension.
- [x] `pyproject.toml` present, declares build dependencies.
- [ ] Compare these files line-by-line with upstream for subtle differences.

### c. CMake Integration
- [x] Top-level `CMakeLists.txt` present.
- [ ] Confirm Python extension is built and installed to correct location by CMake.

---

## 2. Test Clean pip Install
- [ ] Create a fresh virtual environment.
- [ ] Run `pip install .` from the `python/` directory.
- [ ] Verify `ctranslate2` imports and `_ext.*.so` is present in `site-packages`.

---

## 3. Fix Any Build/Packaging Issues
- [ ] If build fails, capture error logs and diagnose.
- [ ] If `.so` not installed, check `setup.py`/CMake install logic.
- [ ] Update packaging or CMake files as needed.

---

## 4. Automate and Document
- [ ] Update `README.md` in root or `python/` with install/build instructions.
- [ ] (Optional) Add GitHub Actions workflow for install/import test.

---

## 5. (Optional) Build and Upload Wheels
- [ ] Build a wheel for the fork.
- [ ] (Optional) Upload to private index or GitHub Release.

---

## Progress
- [x] Initial audit complete
- [ ] Upstream comparison
- [ ] Clean install tested
- [ ] Issues fixed (if any)
- [ ] Documentation/automation
