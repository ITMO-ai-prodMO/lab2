from pathlib import Path

from src.experiment import run_all, save_all_plots, save_csv, save_method_table


def main() -> None:
    rows = run_all(seed=42)
    out_dir = Path("results")
    save_csv(rows, out_dir / "comparison.csv")
    save_all_plots(rows, out_dir)
    save_method_table(rows, out_dir / "method_summary.md")
    print(f"Saved {len(rows)} experiment rows to {out_dir}")


if __name__ == "__main__":
    main()
