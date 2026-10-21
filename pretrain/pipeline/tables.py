"""Paper-ready tables derived only from complete measured seed groups."""

import csv
import io


def write_tables(root, summaries):
    fields = ["task", "size", "objective", "mode", "metric", "mean", "sample_sd", "seeds"]
    rows = []
    for summary in summaries:
        for metric, values in sorted(summary["metrics"].items()):
            rows.append(
                {
                    **{key: summary[key] for key in fields[:4]},
                    "metric": metric,
                    "mean": values["mean"],
                    "sample_sd": values["sample_sd"],
                    "seeds": ",".join(map(str, summary["seeds"])),
                }
            )
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    (root / "paper_tables.csv").write_text(stream.getvalue())
    lines = [
        r"\begin{tabular}{lllllrr}",
        r"Task & Size & Objective & Mode & Metric & Mean & SD \\",
        r"\hline",
    ]
    for row in rows:
        text = [str(row[key]).replace("_", r"\_") for key in fields[:5]]
        text += [f"{row['mean']:.6f}", f"{row['sample_sd']:.6f}"]
        lines.append(" & ".join(text) + r" \\")
    lines.append(r"\end{tabular}")
    (root / "paper_tables.tex").write_text("\n".join(lines) + "\n")
