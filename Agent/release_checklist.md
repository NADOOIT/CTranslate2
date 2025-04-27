# CTranslate2 Fork: Release Checklist & Build Notes

## Ziel
Ein Release (tar.gz oder wheel) meines Forks von CTranslate2 erstellen, das alle Submodules und nativen Komponenten enthält, sodass es direkt via requirements.txt installierbar und als Drop-in-Replacement nutzbar ist (z.B. für faster-whisper).

---

## Schritte zur Release-Erstellung

### 1. Submodules initialisieren
Stelle sicher, dass alle Submodules im Repo enthalten sind:
```sh
git submodule update --init --recursive
```

### 2. Source-Distribution und/oder Wheel bauen
Empfohlen: Beide Formate erzeugen (sdist und wheel):
```sh
cd python
python3 -m pip install --upgrade build setuptools wheel
python3 -m build --sdist --wheel
```
Die gebauten Dateien liegen im `python/dist/`-Verzeichnis.

### 3. Release auf GitHub erstellen
- Erstelle einen neuen Release-Tag (z.B. `vX.Y.Z`).
- Lade die erzeugten Artefakte (`.tar.gz`, `.whl`) im Release-Dialog hoch.

### 4. Test der Installation
In einer frischen venv:
```sh
python3 -m venv testenv
source testenv/bin/activate
pip install https://github.com/NADOOIT/CTranslate2/releases/download/vX.Y.Z/ctranslate2-X.Y.Z.tar.gz
python -c "import ctranslate2; print(ctranslate2.__version__)"
```
Die Ausgabe muss die richtige Version zeigen und der Import darf keinen Fehler werfen.

### 5. Integration in requirements.txt
```txt
ctranslate2 @ https://github.com/NADOOIT/CTranslate2/releases/download/vX.Y.Z/ctranslate2-X.Y.Z.tar.gz
```

---

## Hinweise & Automatisierung
- Für zukünftige Releases empfiehlt sich ein GitHub Actions Workflow, der automatisch Submodules initialisiert, baut und die Artefakte an den Release anfügt.
- Beispiel-Workflow: Siehe offizielle CTranslate2- oder PyPI-Projekte.
- Prüfe, dass im sdist/wheel alle nativen Komponenten und Python-Bindings enthalten sind. Das ctranslate2-Verzeichnis und die .so/.dylib müssen im Paket liegen.

---

## Troubleshooting
- Wenn beim Build aus dem Release kein Python-Binding entsteht: Prüfe setup.py und pyproject.toml auf vollständige Einbindung aller Komponenten.
- Bei Fehlern: Build-Log prüfen, insbesondere auf fehlende Submodules oder native Abhängigkeiten.

---

**Status:**
- [ ] Submodules initialisiert
- [ ] sdist/wheel gebaut
- [ ] Release erstellt und Artefakte hochgeladen
- [ ] Installation aus Release getestet
- [ ] requirements.txt angepasst
- [ ] (Optional) Automatisierung eingerichtet

---

*Zuletzt aktualisiert: 2025-04-27*
