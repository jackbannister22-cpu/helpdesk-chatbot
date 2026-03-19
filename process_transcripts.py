from pathlib import Path
from openai import OpenAI

client = OpenAI()

BASE = Path(".")
RAW = BASE / "raw_transcripts"
CLEANED = BASE / "cleaned_transcripts"
QA = BASE / "qa_blocks"
FLAGGED = BASE / "flagged_for_review"

CLEANING_PROMPT = """
You are preparing a transcript for a private business training knowledge base.

Your job is to CLEAN the transcript, not summarise it.

Rules:
1. Remove greetings, small talk, tech setup, and unrelated chatter.
2. Remove or generalise confidential information:
  - client names
  - company names
  - email addresses
  - phone numbers
  - exact locations
3. Replace identifying details with generic descriptions such as:
  - a plumbing business owner
  - a retail business
  - a service-based business
4. Preserve Christo and Franziska's teaching style.
5. Do not rewrite into generic consultant language.
6. Do not heavily summarise.
7. Output only the cleaned transcript.
8. The transcript may use labels like "Speaker A" and "Speaker B".
9. Assume one speaker is the coach and the other is the client.
10. Identify the coach based on who is giving advice, teaching, explaining concepts, frameworks, strategy, or feedback.
11. Treat the speaker giving strategic guidance and teaching as Christo or Franziska.
12. Preserve the teaching voice and phrasing of Christo and Franziska when cleaning the transcript.
13. If speaker roles are ambiguous, use the content of what is being said to infer who is teaching and who is asking for help.
"""

QA_PROMPT = """
Turn this cleaned transcript into structured knowledge blocks.

Format:

Question:
Answer:
Key Insight:

Rules:
1. Each block should represent one clear business question.
2. Keep the teaching explanation intact.
3. Preserve the tone and language style of Christo and Franziska.
4. Do not invent information.
5. Remove any remaining chatter.
6. Split different topics into separate blocks.
7. Output Q&A blocks only.
"""

def call_ai(prompt, text):
    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}
        ]
    )
    return response.output_text


def process_file(file_path):
    cleaned_file = CLEANED / file_path.name
    qa_file = QA / file_path.name

    # Prevent duplicate processing
    if cleaned_file.exists() and qa_file.exists():
        print("Skipping already processed:", file_path.name)
        return

    print("Processing:", file_path.name)

    try:
        text = file_path.read_text(encoding="utf-8")

        if not text.strip():
            print("File empty → flagged")
            (FLAGGED / file_path.name).write_text(text, encoding="utf-8")
            return

        cleaned = call_ai(CLEANING_PROMPT, text)
        cleaned_file.write_text(cleaned, encoding="utf-8")

        qa = call_ai(QA_PROMPT, cleaned)
        qa_file.write_text(qa, encoding="utf-8")

        print("Finished:", file_path.name)

    except Exception as e:
        print("Error:", file_path.name, "-", str(e))
        FLAGGED.mkdir(exist_ok=True)
        (FLAGGED / file_path.name).write_text(
            text if 'text' in locals() else "",
            encoding="utf-8"
        )


def main():
    CLEANED.mkdir(exist_ok=True)
    QA.mkdir(exist_ok=True)
    FLAGGED.mkdir(exist_ok=True)

    files = list(RAW.glob("*.txt"))

    if not files:
        print("No transcripts found.")
        return

    for file in files:
        process_file(file)

    print("Batch finished.")


if __name__ == "__main__":
    main()