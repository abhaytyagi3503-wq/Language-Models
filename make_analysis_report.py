#!/usr/bin/env python3
"""
make_analysis_report.py

Creates Analysis.pdf from birdKnowledgeTests.csv using matplotlib + reportlab.
"""

from __future__ import annotations
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

HERE = Path(__file__).resolve().parent
TESTS = HERE / "birdKnowledgeTests.csv"
OUT = HERE / "Analysis.pdf"


def save_plot(df, ycol, filename):
    plt.figure(figsize=(8, 5))
    for model_name, group in df.groupby("modelName"):
        g = group.groupby("contextLevel")[ycol].mean().reset_index()
        plt.plot(g["contextLevel"], g[ycol], marker="o", label=model_name)
    plt.xlabel("contextLevel")
    plt.ylabel(ycol)
    plt.title(f"{ycol} vs contextLevel")
    plt.legend()
    plt.tight_layout()
    plt.savefig(HERE / filename, dpi=180)
    plt.close()


def save_plot_workload(df, ycol, filename):
    plt.figure(figsize=(8, 5))
    for model_name, group in df.groupby("modelName"):
        g = group.groupby("numSimultaneousQueries")[ycol].mean().reset_index()
        plt.plot(g["numSimultaneousQueries"], g[ycol], marker="o", label=model_name)
    plt.xlabel("numSimultaneousQueries")
    plt.ylabel(ycol)
    plt.title(f"{ycol} vs workload")
    plt.legend()
    plt.tight_layout()
    plt.savefig(HERE / filename, dpi=180)
    plt.close()


def write_wrapped(c, text, x, y, width=90, line_height=14):
    for line in textwrap.wrap(text, width=width):
        c.drawString(x, y, line)
        y -= line_height
    return y


def main():
    df = pd.read_csv(TESTS)
    save_plot(df, "commonNameErrorRate", "plot_common_context.png")
    save_plot(df, "differenceToThinkingKey", "plot_diff_context.png")
    save_plot_workload(df, "commonNameErrorRate", "plot_common_workload.png")
    save_plot_workload(df, "differenceToThinkingKey", "plot_diff_workload.png")

    c = canvas.Canvas(str(OUT), pagesize=letter)
    width, height = letter
    y = height - 40
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, y, "Assignment 3 Analysis")
    y -= 24
    c.setFont("Helvetica", 11)

    q1 = "Without provided context from relevant articles, how accurate are the responses from the various models? Use the no-context rows (contextLevel=0) to compare error rates."
    q2 = "Did providing the web resource with information about the bird species improve the accuracy of the provided common names and visual attributes? Compare contextLevel=0 against contextLevel=1."
    q3 = "Does submitting fewer queries, each with a larger amount of work, improve or reduce performance? Compare numSimultaneousQueries across 1, 8, 16, 32."

    for txt in [q1, q2, q3]:
        y = write_wrapped(c, txt, 40, y, width=95)
        y -= 10

    # Add images
    for image_name in ["plot_common_context.png", "plot_diff_context.png", "plot_common_workload.png", "plot_diff_workload.png"]:
        if y < 260:
            c.showPage()
            y = height - 40
        c.drawImage(str(HERE / image_name), 40, y - 200, width=520, height=180, preserveAspectRatio=True, mask='auto')
        y -= 220

    c.save()
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
