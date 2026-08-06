/-
Runnable demo of the tight Newton–Schulz operator-norm capstone.

`NewtonSchulzRunnable.newtonSchulz_opNorm` proves: for any Float matrix `X0` passing the single decidable
validity check `matOk X0`, the actual runnable `newtonSchulz X0 (epsDefault …)` (5 Muon iterations, real IEEE
arithmetic) has operator norm `≤ √1.3131 + √(r·c)·rounding` — dimension-free O(1) in `√1.3131 < 1.15`.

`matOk` is a genuine `Bool` the program evaluates. The `#guard_msgs`-checked `#eval`s below RUN it on concrete
matrices — the demo is verified at build time (each guard asserts the evaluated output), so it cannot rot:

  • a well-formed, finite, tall (`r ≤ c`) matrix passes (`true`);
  • a ragged matrix fails (rectangularity);
  • a wrong-orientation (`r > c`) matrix fails (dimension check);
  • a matrix with a NaN entry fails (the Float-comparison finiteness guards);
  • the orthogonalized OUTPUT of `newtonSchulz` on a valid input is itself valid (`matOk … = true`).

So the theorem's sole hypothesis is not an abstract side condition — it is exactly this Boolean, which the
trainer runs on the matrix it already has.
-/
import Puffer.RL.NewtonSchulzRunnable

namespace Puffer.RL.NewtonSchulzDemo

open Puffer.RL.NewtonSchulzRunnable (matOk epsDefault)
open Puffer.FloatR.Muon (newtonSchulz)

-- A well-formed, finite, tall (2×3, r ≤ c) matrix passes the validity check.
/-- info: true -/
#guard_msgs in #eval matOk #[#[1.0, 2.0, 0.5], #[0.0, -1.0, 3.0]]

-- A ragged matrix (second row too short) fails rectangularity.
/-- info: false -/
#guard_msgs in #eval matOk #[#[1.0, 2.0], #[3.0]]

-- A wrong-orientation matrix (3×2, r > c) fails the dimension check.
/-- info: false -/
#guard_msgs in #eval matOk #[#[1.0, 2.0], #[3.0, 4.0], #[5.0, 6.0]]

-- A matrix with a NaN entry (0/0) fails the Float-comparison finiteness guards.
/-- info: false -/
#guard_msgs in #eval matOk #[#[1.0, 2.0, 0.0 / 0.0], #[0.0, 1.0, 2.0]]

-- The orthogonalized output of newtonSchulz on a valid input is itself valid.
/-- info: true -/
#guard_msgs in #eval let X := #[#[1.0, 2.0, 0.5], #[0.0, -1.0, 3.0]];
  matOk (newtonSchulz X (epsDefault X X.size (X[0]!).size))

end Puffer.RL.NewtonSchulzDemo
