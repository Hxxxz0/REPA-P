from pathlib import Path

code = Path(__file__).with_name('main.py').read_text()
exec(compile(code, str(Path(__file__).with_name('main.py')), 'exec'))
