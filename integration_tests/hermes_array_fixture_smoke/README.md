# Hermes Array Fixture Smoke

This fixture validates the real-array diagnostic command chain with an explicit NPZ fixture path.

It is bounded to a tiny local NPZ array fixture and does not use full PDEBench HDF5 data, checkpoints, training, submission builds, or code patching. It exists to prove that Hermes can follow the diagnostic protocol after `inspect-data` is introduced.

