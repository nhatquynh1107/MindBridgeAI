import os
import subprocess
from typing import List, Tuple

def ollama_generate(prompt: str) -> str:
    model = os.getenv("OLLAMA_MODEL", "llama3.2:3b").strip()

    ollama_path = os.getenv("OLLAMA_PATH")
    if not ollama_path:
        ollama_path = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe")

    r = subprocess.run(
        [ollama_path, "run", model, prompt],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or "ollama failed")
    return r.stdout.strip()

def build_prompt(mode: str, instructions: str, history: List[Tuple[str, str]], user_message: str) -> str:
    # Giữ history ngắn để local model không bị “ngợp”
    hist = ""
    for role, content in history[-8:]:
        hist += f"{role.upper()}: {content}\n"

    return (
        f"{instructions}\n\n"
        f"MODE: {mode}\n\n"
        f"CHAT HISTORY:\n{hist}\n"
        f"USER: {user_message}\n"
        f"ASSISTANT:"
    )

def local_generate_reply(mode: str, instructions: str, history: List[Tuple[str, str]], user_message: str) -> str:
    prompt = build_prompt(mode, instructions, history, user_message)
    return ollama_generate(prompt)
