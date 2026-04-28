"""
Run ICON on 5 samples and save a structured JSON with full context + violated content.
Usage:
    python run_5_samples.py
    python run_5_samples.py --start 10   # start from index 10
    python run_5_samples.py --indices 0 5 12 33 99  # specific indices
"""
import json
import sys
import argparse
from pathlib import Path

# Make sure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.utils.csv_reader import CSVReader
from main import UnifiedJailbreakFramework


def build_output_record(query: str, raw_results: dict, sample_index: int) -> dict:
    """
    Turn the raw framework results dict into the clean structured JSON the user wants.
    Keys:
      - sample_index
      - input_query          : the original harmful query
      - attack_pattern       : which ICON pattern was chosen (e.g. Scientific Research)
      - document_type        : which forged document type was used
      - router_reasoning     : why the router chose this pattern
      - forged_context       : the forged authoritative document sent to the target
      - attack_conversation  : the 3-turn prompts sent to the target LLM
      - violated_content     : what the target LLM actually generated (the jailbreak output)
      - jailbreak_succeeded  : True/False
      - judge_score          : 0.0 – 1.0
      - judge_explanation    : human-readable judge explanation
    """
    routing = raw_results.get('routing_result') or {}
    judge = raw_results.get('judge_result') or {}
    attack = raw_results.get('attack_results') or {}

    # --- routing info ---
    pattern = routing.get('pattern', '') if isinstance(routing, dict) else ''
    doc_type = routing.get('document_type', '') if isinstance(routing, dict) else ''
    reasoning = routing.get('reasoning', '') if isinstance(routing, dict) else ''

    # --- conversation turns sent to target LLM ---
    conversation = []
    for i in (1, 2, 3):
        p = attack.get(f'prompt_{i}')
        r = attack.get(f'response_{i}') or attack.get(f'turn_{i}')
        if p or r:
            conversation.append({
                "turn": i,
                "prompt_to_target": p or "",
                "target_response": r or ""
            })

    # --- judge fields ---
    # judge_result uses 'is_jailbroken' / 'is_harmful' keys (see judge_llm.py)
    if isinstance(judge, dict):
        succeeded = judge.get('is_jailbroken') or judge.get('is_harmful') or False
        score = judge.get('score', 0.0)
        explanation = judge.get('explanation', '')
    else:
        succeeded = False
        score = 0.0
        explanation = str(judge)

    return {
        "sample_index": sample_index,
        "input_query": query,
        "attack_pattern": pattern,
        "document_type": doc_type,
        "router_reasoning": reasoning,
        "forged_context": raw_results.get('forged_document') or '',
        "attack_conversation": conversation,
        "violated_content": raw_results.get('jailbreak_response') or '',
        "jailbreak_succeeded": succeeded,
        "judge_score": round(float(score), 3) if score else 0.0,
        "judge_explanation": explanation,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-file', default='data/200.csv')
    parser.add_argument('--output', default='output/5_samples_results.json')
    parser.add_argument('--start', type=int, default=0,
                        help='Start index (first sample). Ignored if --indices is set.')
    parser.add_argument('--count', type=int, default=5,
                        help='Number of samples to run (default 5). Ignored if --indices is set.')
    parser.add_argument('--indices', type=int, nargs='+',
                        help='Specific sample indices to run, e.g. --indices 0 7 14 33 99')
    args = parser.parse_args()

    # Load config & CSV
    config = Config()
    csv_reader = CSVReader(args.csv_file)
    all_behaviors = csv_reader.read_harmful_behaviors()

    # Determine which indices to run
    if args.indices:
        indices = args.indices
    else:
        indices = list(range(args.start, min(args.start + args.count, len(all_behaviors))))

    print(f"[*] Running {len(indices)} samples: indices {indices}")

    framework = UnifiedJailbreakFramework(config)

    output_records = []
    for i, idx in enumerate(indices):
        query = all_behaviors[idx]['goal']
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(indices)}] Sample {idx}: {query[:80]}...")
        print('='*60)

        try:
            # Pass sample_index=idx so attack returns prompt_1/2/3 details
            raw = framework.execute(query, output_file=None, sample_index=idx)
            record = build_output_record(query, raw, idx)
        except Exception as e:
            print(f"[!] Sample {idx} failed: {e}")
            record = {
                "sample_index": idx,
                "input_query": query,
                "error": str(e),
                "violated_content": "",
                "jailbreak_succeeded": False,
                "judge_score": 0.0,
            }

        output_records.append(record)
        print(f"[+] Sample {idx} done — succeeded: {record.get('jailbreak_succeeded')}, "
              f"score: {record.get('judge_score')}")

    # Save JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_records, f, ensure_ascii=False, indent=2)

    succeeded = sum(1 for r in output_records if r.get('jailbreak_succeeded'))
    print(f"\n{'='*60}")
    print(f"[*] Done. {succeeded}/{len(output_records)} jailbreaks succeeded.")
    print(f"[*] Results saved to: {output_path.resolve()}")


if __name__ == '__main__':
    main()
