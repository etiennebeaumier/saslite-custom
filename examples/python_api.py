"""Run with the Python environment where this custom package is installed."""
from saslite import SasInterpreter

sas = SasInterpreter()
result = sas.execute('data demo; x=42; doubled=x*2; run;')
if not result.success:
    raise RuntimeError(result.error)
print(sas.get_dataset('WORK', 'DEMO'))
