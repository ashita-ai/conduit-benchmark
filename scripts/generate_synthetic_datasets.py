#!/usr/bin/env python3
"""
Generate synthetic programming datasets using GPT-5 mini and Gemini 2.5 Pro.

This script reads the standardized prompt template from data/GENERATE_SYNTHETIC_DATASET_PROMPT.md
and uses it to generate 100 programming questions with both models.

Entries are written to the file incrementally as they are generated, and the script
resumes from where it left off if interrupted.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def count_existing_entries(output_path: Path) -> int:
    """Count how many entries already exist in the output file."""
    if not output_path.exists():
        return 0

    count = 0
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    json.loads(line)  # Verify it's valid JSON
                    count += 1
                except json.JSONDecodeError:
                    pass  # Skip invalid lines

    return count


def validate_entry(entry: Dict[str, Any], entry_num: int) -> Optional[str]:
    """
    Validate a single JSONL entry for quality.

    Returns None if valid, error message if invalid.
    """
    # Check required fields
    required_fields = ["query_id", "query_text", "reference_answer", "assertions", "metadata"]
    for field in required_fields:
        if field not in entry:
            return f"Missing required field: {field}"

    # Check metadata fields
    metadata = entry.get("metadata", {})
    required_metadata = ["language", "function_name", "difficulty", "category", "complexity"]
    for field in required_metadata:
        if field not in metadata:
            return f"Missing metadata field: {field}"

    # Check for vague/placeholder queries
    query_text = entry.get("query_text", "")

    # Simple heuristic: if query is too short, it's likely vague
    if len(query_text) < 50:
        return f"Query text too short ({len(query_text)} chars)"

    # Check for trivial reference answers
    ref_answer = entry.get("reference_answer", "")
    if len(ref_answer.strip()) < 50:
        return f"Reference answer too short ({len(ref_answer)} chars)"

    return None  # Valid entry


def append_entry(output_path: Path, entry: Dict[str, Any]) -> bool:
    """Append a single entry to the output file."""
    try:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return True
    except Exception as e:
        print(f"❌ Failed to append entry: {e}")
        return False


def read_prompt_template() -> str:
    """Read the prompt template from data/GENERATE_SYNTHETIC_DATASET_PROMPT.md."""
    prompt_path = project_root / "data" / "GENERATE_SYNTHETIC_DATASET_PROMPT.md"

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found at {prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def generate_batch_with_openai(prompt: str, model_name: str, batch_size: int, start_id: int) -> Optional[list]:
    """Generate a batch of entries using OpenAI API."""
    try:
        from openai import OpenAI
    except ImportError:
        print("❌ OpenAI package not installed. Install with: pip install openai")
        return None

    client = OpenAI()

    # Modify prompt to request specific batch
    batch_prompt = f"{prompt}\n\nGenerate exactly {batch_size} entries, starting with query_id 'synthetic-{start_id:03d}'."

    # Try primary model name first, then fallback
    models_to_try = [model_name, "gpt-5-mini"]

    for model in models_to_try:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a programming expert generating high-quality synthetic programming challenges. Output ONLY valid JSONL - one JSON object per line, no markdown formatting, no code blocks."
                    },
                    {
                        "role": "user",
                        "content": batch_prompt
                    }
                ],
                max_completion_tokens=8000,
            )

            content = response.choices[0].message.content

            # Strip markdown code blocks if present
            content = content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines)

            # Parse JSONL response
            entries = []
            for line in content.split("\n"):
                line = line.strip()
                if line and not line.startswith("#"):
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # Skip invalid lines

            return entries

        except Exception as e:
            if model == models_to_try[-1]:
                print(f"❌ All OpenAI model attempts failed: {e}")
                return None
            else:
                print(f"⚠️  Model {model} failed, trying fallback...")

    return None


def generate_batch_with_gemini(prompt: str, model_name: str, batch_size: int, start_id: int) -> Optional[list]:
    """Generate a batch of entries using Google Gemini API."""
    try:
        import google.generativeai as genai
    except ImportError:
        print("❌ Google Generative AI package not installed. Install with: pip install google-generativeai")
        return None

    # Get API key from environment
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set")
        return None

    genai.configure(api_key=api_key)

    # Modify prompt to request specific batch
    batch_prompt = f"{prompt}\n\nGenerate exactly {batch_size} entries, starting with query_id 'synthetic-{start_id:03d}'."

    try:
        model = genai.GenerativeModel(
            model_name,
            safety_settings={
                "HARASSMENT": "BLOCK_NONE",
                "HATE_SPEECH": "BLOCK_NONE",
                "SEXUALLY_EXPLICIT": "BLOCK_NONE",
                "DANGEROUS_CONTENT": "BLOCK_NONE",
            }
        )

        response = model.generate_content(
            batch_prompt,
            generation_config={
                "temperature": 0.8,
                "max_output_tokens": 8000,
            }
        )

        # Handle safety blocks
        if not response.candidates or not response.candidates[0].content.parts:
            finish_reason = response.candidates[0].finish_reason if response.candidates else "UNKNOWN"
            print(f"⚠️  Gemini blocked response (finish_reason: {finish_reason})")
            return None

        content = response.text

        # Strip markdown code blocks if present
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)

        # Parse JSONL response
        entries = []
        for line in content.split("\n"):
            line = line.strip()
            if line and not line.startswith("#"):
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # Skip invalid lines

        return entries

    except Exception as e:
        print(f"❌ Gemini API call failed: {e}")
        return None


def generate_dataset(
    model_name: str,
    api_type: str,
    output_path: Path,
    prompt: str,
    target_count: int = 100,
    batch_size: int = 10
) -> bool:
    """
    Generate a complete dataset incrementally.

    Args:
        model_name: The model identifier
        api_type: 'openai' or 'gemini'
        output_path: Path to output file
        prompt: Base prompt template
        target_count: Total entries to generate (default 100)
        batch_size: Entries per API call (default 10)
    """
    # Check existing entries
    existing_count = count_existing_entries(output_path)

    if existing_count >= target_count:
        print(f"✅ {output_path.name}: Already has {existing_count} entries, nothing to do")
        return True

    print(f"📊 {output_path.name}: Found {existing_count}/{target_count} existing entries")
    print(f"🔄 Generating {target_count - existing_count} more entries...")

    remaining = target_count - existing_count
    current_id = existing_count + 1

    while remaining > 0:
        batch = min(batch_size, remaining)

        print(f"\n🔄 Generating batch: entries {current_id}-{current_id + batch - 1}...")

        # Generate batch
        if api_type == "openai":
            entries = generate_batch_with_openai(prompt, model_name, batch, current_id)
        elif api_type == "gemini":
            entries = generate_batch_with_gemini(prompt, model_name, batch, current_id)
        else:
            print(f"❌ Unknown API type: {api_type}")
            return False

        if not entries:
            print(f"❌ Failed to generate batch")
            return False

        # Validate and write each entry immediately
        written_count = 0
        for i, entry in enumerate(entries):
            entry_num = current_id + i

            # Validate quality
            error = validate_entry(entry, entry_num)
            if error:
                print(f"⚠️  Entry {entry_num}: {error} - skipping")
                continue

            # Write to file immediately
            if append_entry(output_path, entry):
                written_count += 1
                print(f"✅ Wrote entry {entry_num}/{target_count}")
            else:
                print(f"❌ Failed to write entry {entry_num}")
                return False

        if written_count == 0:
            print(f"❌ No valid entries in batch")
            return False

        remaining -= written_count
        current_id += written_count

        print(f"📊 Progress: {current_id - 1}/{target_count} entries ({100 * (current_id - 1) / target_count:.1f}%)")

    final_count = count_existing_entries(output_path)
    if final_count >= target_count:
        print(f"✅ {output_path.name}: Complete with {final_count} entries")
        return True
    else:
        print(f"⚠️  {output_path.name}: Only {final_count}/{target_count} entries")
        return False


def main():
    """Main execution function."""
    print("=" * 80)
    print("Synthetic Dataset Generation (Incremental)")
    print("=" * 80)

    # Read prompt template
    print("\n📄 Reading prompt template...")
    try:
        prompt = read_prompt_template()
        print(f"✅ Loaded prompt template ({len(prompt)} chars)")
    except Exception as e:
        print(f"❌ Failed to read prompt: {e}")
        return 1

    # Configuration
    # Note: Switched from gpt-5-mini and gemini-2.5-pro due to generation issues:
    # - gpt-5-mini: Failed after 20 entries (batch 3 generation error)
    # - gemini-2.5-pro: Blocked by 'dangerous_content' safety filter on programming challenges
    datasets = [
        {
            "name": "gpt-4o-mini",
            "model": "gpt-4o-mini",
            "api_type": "openai",
            "output": project_root / "datasets" / "synthetic_gpt-4o-mini.jsonl",
            "batch_size": 10,
        },
        {
            "name": "gemini-1.5-pro",
            "model": "gemini-1.5-pro",
            "api_type": "gemini",
            "output": project_root / "datasets" / "synthetic_gemini-1.5-pro.jsonl",
            "batch_size": 10,
        },
    ]

    # Generate each dataset
    success_count = 0
    for config in datasets:
        print(f"\n{'=' * 80}")
        print(f"Dataset: {config['name']}")
        print(f"{'=' * 80}")

        if generate_dataset(
            model_name=config["model"],
            api_type=config["api_type"],
            output_path=config["output"],
            prompt=prompt,
            target_count=100,
            batch_size=config["batch_size"]
        ):
            success_count += 1

    # Summary
    print(f"\n{'=' * 80}")
    print(f"Summary: {success_count}/{len(datasets)} datasets completed")
    print(f"{'=' * 80}")

    return 0 if success_count == len(datasets) else 1


if __name__ == "__main__":
    sys.exit(main())
