"""Read independent repetition progress without waiting or starting any work."""
import argparse
from pathlib import Path
import statistics

from experiments.baselines.b4_embodiskill.state import read_json, write_json
from .authority import METHODS, assert_source_unchanged, file_hash


def collect(root):
    audit=read_json(root/"source_audit.json")
    methods={}
    for method in METHODS:
        original=audit["sources"][method]
        assert_source_unchanged(original)
        rounds=[dict(repeat=0,**original["original_result"])]
        pending=[]
        for repeat in (1,2):
            output=root/f"{method}_repeat_{repeat}"
            if not (output/"completion.json").exists():
                progress=read_json(output/"progress.json") if (output/"progress.json").exists() else {}
                pending.append(dict(repeat=repeat,progress=progress,output=str(output)))
                continue
            completion=read_json(output/"completion.json")
            if not completion["passed"] or file_hash(output/"test_report.json")!=completion["report_sha256"]:
                raise ValueError("Repeat completion/report integrity failed")
            report=read_json(output/"test_report.json")
            if report["engineering_smoke"] or report["frozen_digest"]!=original["frozen_digest"]:
                raise ValueError("Cannot combine smoke or different frozen assets with formal repetitions")
            rounds.append(dict(repeat=repeat,tasks=report["test"]["tasks"],
                official=report["test"]["official_success"],strict=report["test"]["common_strict_success"],
                successful_response_tokens_per_task=report["successful_response_usage"]["total_tokens"]/report["test"]["tasks"]))
        entry=dict(complete=len(rounds)==3,rounds=rounds,pending=pending)
        if entry["complete"]:
            entry["mean_std"]={}
            for metric in ("official","strict"):
                values=[r[metric]/r["tasks"] for r in rounds]
                entry["mean_std"][metric]=dict(mean=statistics.mean(values),std=statistics.stdev(values),ddof=1)
        methods[method]=entry
    return dict(methods=methods,statistical_unit="three_inference_repetitions_of_one_seed42_frozen_asset",
                independent_training_seeds=1,complete=all(v["complete"] for v in methods.values()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output")
    p.add_argument("--write-report",action="store_true")
    args=p.parse_args()
    root=Path(args.output).absolute()
    result=collect(root)
    for method,entry in result["methods"].items():
        print(method,[(r["repeat"],r["official"],r["strict"],r["tasks"]) for r in entry["rounds"]],
              "pending:",[(r["repeat"],r["progress"]) for r in entry["pending"]])
    if args.write_report:
        write_json(root/"repeat_comparison.json",result)
        print(root/"repeat_comparison.json")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
