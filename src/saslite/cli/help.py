"""Versioned usage guide with live CLI, procedure and function inventories."""
import ast
import inspect
import json
from pathlib import Path
import sys
import textwrap

PACKAGE = Path(__file__).resolve().parents[1]


def help_data():
    return json.loads((Path(__file__).with_name('help_data.json')).read_text(encoding='utf-8'))


def supported_procedures():
    # PROC SQL is dispatched via its own AST; the rest use the registration list.
    facade = (PACKAGE / 'api/facade.py').read_text(encoding='utf-8')
    names = {'SQL'}
    for node in ast.walk(ast.parse(facade)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'register_proc' and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            names.add(node.args[0].value.upper())
    return sorted(names)


def validate_reference():
    data = help_data()
    documented = set(data['procedures'])
    actual = set(supported_procedures())
    if documented != actual:
        raise ValueError(f'Update help procedure coverage: missing={sorted(actual-documented)}, stale={sorted(documented-actual)}')
    for topic, entry in {**data['topics'], **data['procedures']}.items():
        if not entry.get('title') or not entry.get('body'):
            raise ValueError(f'Help topic {topic} is empty')


def _section(title, body):
    lines = [title, '-' * len(title)]
    for line in body.splitlines():
        if not line.strip() or line.startswith('    '):
            lines.append(line)
        else:
            lines.append(textwrap.fill(line, width=80))
    return '\n'.join(lines) + '\n'


def _function_help(name=None):
    from saslite.functions import build_default_registry
    registry = build_default_registry()
    if name:
        function = registry.get(name)
        if function is None:
            raise ValueError(f'Unknown function: {name}')
        return _section(name.upper() + str(inspect.signature(function)),
                        inspect.getdoc(function) or 'Registered SASLite function.')
    return _section('Built-in functions (from the installed registry)',
                    ', '.join(registry.names) + '\n\nUse: saslite-custom help function SQRT\n'
                    'Signatures describe the Python implementation; SAS calls use ordinary parentheses.\n'
                    'Example: data demo; x=sqrt(16); y=mean(1,2,3); run;')


def render_help(topic, parser):
    from saslite import __version__
    data = help_data()
    topic = topic.strip().lower()
    aliases = data['aliases']
    topic = aliases.get(topic, topic)
    if topic.startswith('proc '):
        topic = topic[5:].strip()
    if topic.startswith('function '):
        return _function_help(topic[9:].strip())
    header = f'SASLite Custom {__version__} — Usage Guide\n'
    topics = ', '.join(['all', 'options', 'procedures', 'functions', *data['topics']])
    index = _section('Find help', 'saslite-custom help                  Full guide\n'
                     'saslite-custom help topics           Topic index\n'
                     'saslite-custom help libraries        One topic\n'
                     'saslite-custom help proc print       One procedure\n'
                     'saslite-custom --help                Command options\n\n'
                     'Topics: ' + topics + '\n'
                     'Procedure names, mww and fp also work as topics.')
    if topic in ('topics', 'index'):
        return header + '\n' + index
    if topic == 'options':
        return header + '\n' + parser.format_help()
    if topic == 'functions':
        return header + '\n' + _function_help()
    if topic.upper() in data['procedures']:
        entry = data['procedures'][topic.upper()]
        return header + '\n' + _section(entry['title'], entry['body'])
    if topic in data['topics']:
        entry = data['topics'][topic]
        return header + '\n' + _section(entry['title'], entry['body'])
    if topic not in ('all', 'procedures'):
        raise ValueError(f'Unknown help topic: {topic}. Run saslite-custom help topics.')
    sections = [header]
    if topic == 'all':
        sections.extend([index, parser.format_help()])
        sections.extend(_section(entry['title'], entry['body']) for entry in data['topics'].values())
    sections.append(_section('Procedure index', ', '.join(supported_procedures()) +
                            '\nThese procedures implement subsets of SAS syntax, not all SAS options.'))
    sections.extend(_section('PROC ' + name + ' — ' + entry['title'], entry['body'])
                    for name, entry in data['procedures'].items())
    if topic == 'all':
        sections.append(_function_help())
    return '\n'.join(sections)


def show_help(topic, parser):
    try:
        print(render_help(topic, parser))
    except ValueError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2
    return 0
