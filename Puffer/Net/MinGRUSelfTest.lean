/- Standalone `main` wrapper around `Puffer.Net.MinGRU.emitSelfTest`, kept OUT of the importable
   library so the MinGRU module can be imported by the trainer without a `main` clash. Run with
   `lean --run Puffer/Net/MinGRUSelfTest.lean`; `tools/mingru_ref.py` recomputes and checks. -/
import Puffer.Net.MinGRU

def main : IO Unit := Puffer.Net.MinGRU.emitSelfTest
