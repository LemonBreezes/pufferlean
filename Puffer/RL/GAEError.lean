/-
First real→bf16 error bound on an actual RL quantity.

The GAE/V-Trace kernel reads its inputs (rewards, values, …) from bf16 buffers, so
in effect it evaluates the advantage recurrence on bf16-rounded TD errors. Here we
bound how far that pulls the computed advantage from the exact real-number GAE:

    |gaeHead w ds − gaeHead w (map bf16 ds)| ≤ 2^{-7} · Σ_i w^i · |ds_i|

i.e. the deviation is at most the bf16 unit roundoff times the (absolutely
weighted) discounted magnitude of the TD errors. For `0 ≤ w < 1` and TD errors
bounded by `M`, the RHS is `≤ 2^{-7} · M / (1 − w)` — a uniform, provable error
bar linking the real spec to its bf16 approximation. This is the model↔float loop
closed on a genuine trainer quantity for the first time.
-/
import Puffer.Numeric.Bf16
import Puffer.RL.GAE

namespace Puffer.RL.GAE

open Finset Puffer.Numeric

/-- **GAE bf16 error bound.** Evaluating the advantage recurrence on bf16-rounded
    TD errors deviates from the exact real GAE by at most `2^{-7}·Σ w^i·|δ_i|`. -/
theorem gaeHead_bf16_error (w : ℝ) (hw : 0 ≤ w) (ds : List ℝ) :
    |gaeHead w ds - gaeHead w (ds.map bf16)|
      ≤ ∑ i ∈ range ds.length, w ^ i * ((2 : ℝ) ^ (-7 : ℤ) * |ds.getD i 0|) := by
  induction ds with
  | nil => simp
  | cons d ds' ih =>
      rw [List.map_cons, gaeHead_cons, gaeHead_cons, List.length_cons, sum_range_succ']
      simp only [List.getD_cons_succ, List.getD_cons_zero, pow_zero, one_mul, pow_succ]
      -- triangle inequality, keeping the recursive term first to match the sum order
      have htri :
          |(d + w * gaeHead w ds') - (bf16 d + w * gaeHead w (ds'.map bf16))|
            ≤ w * |gaeHead w ds' - gaeHead w (ds'.map bf16)| + |d - bf16 d| := by
        calc |(d + w * gaeHead w ds') - (bf16 d + w * gaeHead w (ds'.map bf16))|
            = |w * (gaeHead w ds' - gaeHead w (ds'.map bf16)) + (d - bf16 d)| := by congr 1; ring
          _ ≤ |w * (gaeHead w ds' - gaeHead w (ds'.map bf16))| + |d - bf16 d| := abs_add_le _ _
          _ = w * |gaeHead w ds' - gaeHead w (ds'.map bf16)| + |d - bf16 d| := by
                rw [abs_mul, abs_of_nonneg hw]
      -- per-element bf16 bound on the head; recursive bound on the tail
      have hd : |d - bf16 d| ≤ (2 : ℝ) ^ (-7 : ℤ) * |d| := by
        rw [abs_sub_comm]; exact bf16_error_bound d
      have hrec := mul_le_mul_of_nonneg_left ih hw
      rw [mul_sum] at hrec
      calc |(d + w * gaeHead w ds') - (bf16 d + w * gaeHead w (ds'.map bf16))|
          ≤ w * |gaeHead w ds' - gaeHead w (ds'.map bf16)| + |d - bf16 d| := htri
        _ ≤ (∑ i ∈ range ds'.length, w * (w ^ i * ((2 : ℝ) ^ (-7 : ℤ) * |ds'.getD i 0|)))
              + (2 : ℝ) ^ (-7 : ℤ) * |d| := add_le_add hrec hd
        _ = (∑ i ∈ range ds'.length, w ^ i * w * ((2 : ℝ) ^ (-7 : ℤ) * |ds'.getD i 0|))
              + (2 : ℝ) ^ (-7 : ℤ) * |d| := by
              congr 1
              apply sum_congr rfl
              intro i _
              ring

end Puffer.RL.GAE
