import re
import sys
from pathlib import Path

def refactor_file(path):
    content = Path(path).read_text()

    # Map-Typ ersetzen
    content = re.sub(
        r'std::unordered_map<\s*std::string\s*,\s*SpeakerProfile\s*>',
        'std::unordered_map<std::string, std::unique_ptr<SpeakerProfile>>',
        content
    )
    # Insertions ersetzen
    content = re.sub(
        r'_active_profiles\.insert_or_assign\(([^,]+),\s*([^)]+)\)',
        r'_active_profiles.insert_or_assign(\1, std::make_unique<SpeakerProfile>(\2))',
        content
    )
    content = re.sub(
        r'_active_profiles\.try_emplace\(([^,]+),\s*([^)]+)\)',
        r'_active_profiles.try_emplace(\1, std::make_unique<SpeakerProfile>(\2))',
        content
    )
    # Dereferenzierung für Funktionsaufrufe
    content = re.sub(
        r'process_with_model\(([^,]+),\s*([^)]+),\s*_active_profiles\[([^\]]+)\]\)',
        r'process_with_model(\1, \2, *( _active_profiles[\3]))',
        content
    )
    # Insert mit std::move entfernen, falls vorhanden
    content = re.sub(
        r'std::make_unique<SpeakerProfile>\(std::move\(([^)]+)\)\)',
        r'std::make_unique<SpeakerProfile>(\1)',
        content
    )

    Path(path).write_text(content)
    print(f"Refactored {path}")

if __name__ == "__main__":
    for filename in sys.argv[1:]:
        refactor_file(filename)
