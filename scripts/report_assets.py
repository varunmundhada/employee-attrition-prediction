"""Copies figures into report/figures and writes report/generated/*.tex from the saved metrics,
so the LaTeX report always shows the numbers of the latest training run."""

import json
import shutil

from src import config

REPORT = config.ROOT / "report"
GEN = REPORT / "generated"


def esc(s):
    return str(s).replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


def main():
    GEN.mkdir(parents=True, exist_ok=True)
    (REPORT / "figures").mkdir(exist_ok=True)
    for png in config.FIGURES_DIR.glob("*.png"):
        shutil.copy(png, REPORT / "figures" / png.name)
    shots = config.ROOT / "docs" / "screenshots"
    if shots.exists():
        for png in shots.glob("*.png"):
            shutil.copy(png, REPORT / "figures" / f"app_{png.name}")

    m = json.loads(config.METRICS_PATH.read_text())
    eda = json.loads((config.MODEL_DIR / "eda_summary.json").read_text())
    best = m["best_model"]
    b = m["models"][best]
    t = b["test"]
    cm = t["confusion"]
    by = eda["by"]
    ot = {r["grp"]: r for r in by["OverTime"]}
    ms = {r["grp"]: r for r in by["MaritalStatus"]}
    role = by["JobRole"]

    numbers = {
        "BestModel": best,
        "BestCVAUC": f"{b['cv_auc']:.3f}",
        "BestOOFAUC": f"{b['oof_auc']:.3f}",
        "BestTestAUC": f"{b['test_auc']:.3f}",
        "BestAcc": f"{100 * t['accuracy']:.1f}",
        "BestPrec": f"{100 * t['precision']:.1f}",
        "BestRecall": f"{100 * t['recall']:.1f}",
        "BestFone": f"{100 * t['f1']:.1f}",
        "BestThr": f"{t['threshold']:.2f}",
        "BestRecallHalf": f"{100 * b['test_at_0.5']['recall']:.0f}",
        "BestPrecHalf": f"{100 * b['test_at_0.5']['precision']:.0f}",
        "CmTN": cm["tn"], "CmFP": cm["fp"], "CmFN": cm["fn"], "CmTP": cm["tp"],
        "TestLeavers": cm["tp"] + cm["fn"], "TestRows": sum(cm.values()),
        "DevRows": m["split"]["development"]["rows"], "DevLeavers": m["split"]["development"]["leavers"],
        "TestRowsSplit": m["split"]["test"]["rows"],
        "AttrRate": f"{eda['attrition_rate']:.1f}", "Leavers": eda["leavers"],
        "OTYes": f"{ot['Yes']['rate']:.1f}", "OTNo": f"{ot['No']['rate']:.1f}",
        "SingleRate": f"{ms['Single']['rate']:.1f}", "MarriedRate": f"{ms['Married']['rate']:.1f}",
        "TopRole": esc(role[0]["grp"]), "TopRoleRate": f"{role[0]['rate']:.1f}",
        "LowRole": esc(role[-1]["grp"]), "LowRoleRate": f"{role[-1]['rate']:.1f}",
        "TopFeature": esc(m["feature_importance"][0]["feature"].split(" = ")[0]),
    }
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in numbers.items()]
    (GEN / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf8")

    rows = []
    for name, r in m["models"].items():
        bold = (lambda s: f"\\textbf{{{s}}}") if name == best else (lambda s: s)
        rows.append(" & ".join(bold(x) for x in [
            name, f"{r['cv_auc']:.3f}", f"{r['oof_auc']:.3f}", f"{r['test_auc']:.3f}",
            f"{r['test']['accuracy']:.3f}", f"{r['test']['precision']:.3f}", f"{r['test']['recall']:.3f}",
            f"{r['test']['f1']:.3f}", f"{r['test']['threshold']:.2f}"]) + r" \\ \hline")
    (GEN / "model_table.tex").write_text("\n".join(rows) + "\n", encoding="utf8")

    rows = []
    for name, r in m["models"].items():
        params = ", ".join(f"{k} = {v:g}" if isinstance(v, (int, float)) else f"{k} = {v}"
                           for k, v in r["best_params"].items())
        rows.append(f"{name} & {esc(params)} & {r['train_seconds']:.0f} \\\\ \\hline")
    (GEN / "params_table.tex").write_text("\n".join(rows) + "\n", encoding="utf8")

    rows = []
    for name, r in m["models"].items():
        a = r["test_at_0.5"]
        rows.append(f"{name} & {a['precision']:.3f} & {a['recall']:.3f} & {a['f1']:.3f} & "
                    f"{r['test']['threshold']:.2f} & {r['test']['recall']:.3f} & {r['test']['f1']:.3f} \\\\ \\hline")
    (GEN / "threshold_table.tex").write_text("\n".join(rows) + "\n", encoding="utf8")

    rows = [f"{i} & {esc(d['feature'])} & {d['importance']:.4f} \\\\ \\hline"
            for i, d in enumerate(m["feature_importance"][:10], 1)]
    (GEN / "importance_table.tex").write_text("\n".join(rows) + "\n", encoding="utf8")

    scale = config.MODEL_DIR / "scale_benchmark.json"
    if scale.exists():
        rows = [f"{r['rows']:,} & {r['model']} & {r['fit_seconds']:.1f} & {r['score_seconds']:.1f} & "
                f"{r['auc']:.3f} \\\\ \\hline" for r in json.loads(scale.read_text())]
        (GEN / "scale_table.tex").write_text("\n".join(rows).replace(",", "{,}") + "\n", encoding="utf8")
    else:
        (GEN / "scale_table.tex").write_text(
            r"Not run yet & & & & \\ \hline" + "\n", encoding="utf8")
    print("Report assets written to", GEN)


if __name__ == "__main__":
    main()
