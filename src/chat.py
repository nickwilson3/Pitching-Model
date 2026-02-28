import sys
import os

from agents import AGENTS

USAGE = """
Usage: python src/chat.py [agent]

Available agents:
  scientist  — Data Scientist: baseball ML expert, model design and feature selection
  engineer   — Data Engineer: ETL pipelines, DuckDB schema, baseball API ingestion
  analyst    — Data Analyst: day-of operations, lineup confirmation, weather

Example:
  python src/chat.py scientist
""".strip()


def main():
    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(1)

    agent_key = sys.argv[1].lower()

    if agent_key not in AGENTS:
        print(f"Unknown agent: '{agent_key}'\n")
        print(USAGE)
        sys.exit(1)

    agent_class = AGENTS[agent_key]
    agent = agent_class()
    agent.run_interactive()


if __name__ == "__main__":
    main()
