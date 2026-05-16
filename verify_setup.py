import os
import sys
from importlib import import_module

REQUIRED_PACKAGES = [
    "requests",
    "bs4",
    "pandas",
    "numpy",
    "sentence_transformers",
    "chromadb",
    "rank_bm25",
    "groq",
    "langchain",
    "langchain_groq",
    "ragas",
    "datasets",
    "dotenv",
    "tqdm",
]


def check_python_version():
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 10):
        print(f"  [FAIL] Python {major}.{minor} detected. Python 3.10+ required.")
        return False
    print(f"  [OK] Python {major}.{minor}")
    return True


def check_packages():
    all_ok = True
    for pkg in REQUIRED_PACKAGES:
        try:
            import_module(pkg)
            print(f"  [OK] {pkg}")
        except ImportError as e:
            print(f"  [FAIL] {pkg} — {e}")
            all_ok = False
    return all_ok


def check_groq_api():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("  [FAIL] python-dotenv not installed")
        return False

    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_GENERATION_MODEL", "llama-3.3-70b-versatile")

    if not api_key or api_key == "your_groq_api_key_here":
        print("  [FAIL] GROQ_API_KEY not set in .env")
        return False

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly: pong"}],
            max_tokens=10,
            temperature=0,
        )
        reply = response.choices[0].message.content.strip()
        print(f"  [OK] Groq API reachable. Model: {model}. Reply: {reply!r}")
        return True
    except Exception as e:
        print(f"  [FAIL] Groq API call failed — {e}")
        return False


def main():
    print("\n=== Python version ===")
    py_ok = check_python_version()

    print("\n=== Required packages ===")
    pkg_ok = check_packages()

    print("\n=== Groq API connectivity ===")
    api_ok = check_groq_api()

    print("\n=== Summary ===")
    if py_ok and pkg_ok and api_ok:
        print("  All checks passed. Environment is ready.")
        sys.exit(0)
    else:
        print("  One or more checks failed. Fix the issues above before continuing.")
        sys.exit(1)


if __name__ == "__main__":
    main()