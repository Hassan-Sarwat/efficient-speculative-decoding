
def get_existing_instructions(filepath: Path) -> set:
    """Reads a JSONL file and returns a set of existing instructions (questions) to avoid duplicates."""
    existing = set()
    if filepath.exists():
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        if "instruction" in data:
                            existing.add(data["instruction"])
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.warning(f"Error reading existing file {filepath}: {e}")
    return existing
