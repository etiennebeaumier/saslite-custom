"""External library assignments so existing SAS source needs no path edits."""
from pathlib import Path
import json
import re

from saslite.storage.sas_backend import SasBackend


def library_mappings(values, base):
    if not isinstance(values, dict):
        raise ValueError('libraries must be an object mapping SAS librefs to directories')
    result = {}
    for name, folder in values.items():
        name = str(name).upper()
        if not re.fullmatch(r'[A-Z_][A-Z0-9_]{0,7}', name) or name == 'WORK':
            raise ValueError(f'Invalid or reserved library reference: {name}')
        if not isinstance(folder, str) or not folder:
            raise ValueError(f'Library {name} requires a directory path')
        path = Path(folder).expanduser()
        path = (Path(base) / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_dir():
            raise ValueError(f'Library {name}: directory does not exist: {path}')
        result[name] = str(path)
    return result


def configure_project(session, reporter, script_dir):
    """Load only the script's adjacent config; explicit CLI mappings win."""
    base = Path(script_dir).resolve()
    config = base / '.saslite.json'
    mappings = {}
    if config.is_file():
        data = json.loads(config.read_text(encoding='utf-8'))
        if not isinstance(data, dict) or set(data) - {'libraries'}:
            raise ValueError(f'{config}: expected an object with a libraries field')
        mappings = library_mappings(data.get('libraries', {}), base)
    mappings.update(session.get_option('CLI_LIBRARIES', {}))
    # Validate all mappings before changing session state.
    old = session.get_option('PROJECT_LIBRARIES', {})
    for name, backend in old.items():
        if session.storage.get_backend(name) is backend:
            del session.storage._backends[name]
    assigned = {}
    for name, folder in mappings.items():
        backend = SasBackend(folder, libref=name, format='xpt')
        session.storage.register(name, backend)
        assigned[name] = backend
        reporter.note(f'Library {name} assigned to {folder} from project/CLI configuration')
    session.set_option('PROJECT_LIBRARIES', assigned)
    session.set_option('LIBRARY_OVERRIDES', mappings)
    session.set_option('SOURCE_DIR', str(base))
