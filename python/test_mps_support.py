import ctranslate2
import sys

print("CTranslate2 Version:", ctranslate2.__version__)

# Test, ob MPS als Compute-Type unterstützt wird (funktioniert nur, wenn MPS wirklich eingebaut wurde)
try:
    mps_types = ctranslate2.get_supported_compute_types("mps")
    print("get_supported_compute_types('metal'):", mps_types)
except Exception as e:
    print("get_supported_compute_types('metal') nicht verfügbar oder fehlgeschlagen:", e)

# Teste, ob ein Translator mit MPS-Device erzeugt werden kann
try:
    # Ersetze den Modellpfad ggf. durch ein echtes Modell, falls du eins hast
    translator = ctranslate2.Translator("dummy-model", device="metal")
    print("Translator mit device='metal' konnte erzeugt werden!")
except Exception as e:
    print("Translator mit device='metal' konnte NICHT erzeugt werden:", e)

# Teste, ob ein Generator mit MPS-Device erzeugt werden kann
try:
    generator = ctranslate2.Generator("dummy-model", device="metal")
    print("Generator mit device='metal' konnte erzeugt werden!")
except Exception as e:
    print("Generator mit device='metal' konnte NICHT erzeugt werden:", e)
