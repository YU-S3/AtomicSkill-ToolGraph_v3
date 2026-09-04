"""Common experiment infrastructure shared by all baseline methods.

Only manifest/model/usage/trace/freeze/authority plumbing lives here.  This
package deliberately imports no Ours knowledge algorithm (Planner, Extractor,
Admission, Lifecycle, EvidenceLedger); the strict post-evaluator is the sole
consumer of the Ours Harness boundary and never feeds facts back to a
baseline agent.
"""
