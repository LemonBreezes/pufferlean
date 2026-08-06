import Puffer.RL.AdvNormPerturb
#print axioms Puffer.RL.AdvNormPerturb.advNorm_perturb

def advNormF (a m v e : Float) : Float := (a - m)/(Float.sqrt v + e)
def lhs (a1 m1 v1 a2 m2 v2 e : Float) : Float := Float.abs (advNormF a1 m1 v1 e - advNormF a2 m2 v2 e)
def rhs (a1 m1 v1 a2 m2 v2 e : Float) : Float :=
  Float.abs (a1-m1) * Float.sqrt (Float.abs (v1-v2)) / ((Float.sqrt v1 + e)*(Float.sqrt v2 + e))
  + (Float.abs (a1-a2) + Float.abs (m1-m2))/(Float.sqrt v2 + e)
#eval (lhs 2 0 4 1 0 1 1, rhs 2 0 4 1 0 1 1)
#eval (lhs 3 1 4 1 0 4 1, rhs 3 1 4 1 0 4 1)
-- adversarial: negative-heavy, small eps
#eval (lhs (-5) 2 0.01 3 (-4) 9 0.001, rhs (-5) 2 0.01 3 (-4) 9 0.001)
