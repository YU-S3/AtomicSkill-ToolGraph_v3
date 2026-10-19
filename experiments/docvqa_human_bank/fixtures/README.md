# Synthetic fixtures only

`synthetic_form.png` was generated for local interface tests. It is not a DocVQA dataset image, does not contain benchmark answers, and has no connection to official evaluator records. The three JSON files mock Dynamic node *outputs* so that the graph's actual field references and authored pure Python computations can be checked without an LLM or Docker. Real graph success requires the user's model to return those values from the actual image.
