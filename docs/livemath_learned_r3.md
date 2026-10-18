# LiveMath learned program R3

Target agreed before execution: 70–80 correct out of the canonical Test100 counts
as close to the human R3.1 reference. This is a development-exposed comparison,
not a new blind test or a guarantee of reaching the target.

P7-R2 only learned text Skills. Its Train12 feedback improved 2→4 and Val17
improved 3→4, but no meta-option task improved. All 22 such tasks in the original
Train60 were incorrect and none chose the meta option. That supports investigating
a failure pattern, not asserting a correct theorem or fabricating positive traces.

R3 extends the P7 source, Provider, budget and evaluator modules. The learner can
propose a public-choice predicate alongside a text Skill. A generic compiler turns
that model-produced predicate into a finite Python Program. There are no built-in
meta-option matching terms, answer labels, task IDs, or human-Bank assets in this
compiler. Predicates are empirical benchmark priors, never mathematical proofs.

The Program receives only the complete public choices. It selects only a unique
match, abstains on none/multiple matches, and runs in the pinned Docker Worker with
no network or tool calls. Source provenance, actual execution, label/order
invariance and ambiguous-input abstention are verified before any real takeover.
Competing Programs must agree; a conflict returns to the ordinary solver.

All original Train60 traces are reused without solving them again for induction.
Host-only official Train scoring checks each predicate across Train60. Raw gold
labels are never sent to the learner. There are at most three candidates, two
substantive edit rounds, and one structural repair per learner request. Existing
Train12 is development feedback, not an independent holdout. Existing NoSkill
responses may be reused only after exact input/Runtime/score/usage verification.

Candidate admission requires at least four Train dispatches, at least 80% correct
on those dispatches and no regression on historically correct source tasks. These
are in-sample checks. Paired Train12 then selects the policy; Val17 evaluates that
fixed policy. The initial finite cycle does not edit candidates from Val labels.
Frozen publication requires paired Val net gain >=2, losses <=1, no increase in
empty answers and at least 12/17 correct. Failure remains a real negative result.
At least two net gains must also belong to actual Program or prompt interventions;
same-payload sampling differences cannot qualify a Skill. Reused NoSkill responses
are historical comparisons with matched inputs and parameters, not simultaneous
randomized solves, and their original timestamps and charges remain visible.

The Runtime retains P7's 65,536-token high-thinking main request, 600s timeout,
128,000-token parent budget and one bounded 16,384-token low-thinking finalizer.
Program dispatches use zero model HTTP and are recorded separately from model
answers. Mathematical and benchmark-prior performance are reported separately.

All historical fees and unresolved-request reservations remain in the 45M-token /
9,165-HTTP cumulative ledger. Stage caps and exact source identities are frozen
before execution. No stage borrows another stage's allowance. Test is a separate
command requiring a successful Val review and a frozen Bank. It never triggers
learning, repair, candidate reselection, or an automatic second Test attempt.
