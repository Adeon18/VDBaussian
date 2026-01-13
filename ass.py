import slangpy as spy
from pathlib import Path

device = spy.Device(enable_debug_layers=True)
cmd = device.create_command_encoder()

print("\n[DEBUG] Methods on CommandEncoder:")
print([m for m in dir(cmd) if not m.startswith('_')])