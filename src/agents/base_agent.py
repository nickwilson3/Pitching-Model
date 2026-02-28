import os
import anthropic
from dotenv import load_dotenv


class BaseAgent:
    AGENT_NAME: str = "Base Agent"
    AGENT_ROLE: str = "Base agent role"
    SYSTEM_PROMPT: str = "You are a helpful assistant."
    MODEL: str = "claude-opus-4-6"
    MAX_TOKENS: int = 8096

    def __init__(self):
        load_dotenv()
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. "
                "Copy .env.example to .env and add your key."
            )
        self.client = anthropic.Anthropic(api_key=api_key)
        self.conversation_history: list[dict] = []

    def chat(self, user_message: str) -> str:
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
        })
        try:
            response = self.client.messages.create(
                model=self.MODEL,
                max_tokens=self.MAX_TOKENS,
                system=self.SYSTEM_PROMPT,
                messages=self.conversation_history,
            )
            assistant_message = response.content[0].text
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message,
            })
            return assistant_message
        except anthropic.APIError as e:
            self.conversation_history.pop()  # Remove the user turn — keep history clean
            return f"[API Error: {e}]"

    def run_interactive(self) -> None:
        separator = "-" * 60
        print(separator)
        print(f"  {self.AGENT_NAME}")
        print(f"  {self.AGENT_ROLE}")
        print(separator)
        print("Type your message and press Enter.")
        print("Type 'exit' or 'quit' to end the session.")
        print(separator)

        try:
            while True:
                user_input = input("\nYou: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit"):
                    print(f"\n{self.AGENT_NAME}: Ending session. Good luck with the model.")
                    break
                response = self.chat(user_input)
                print(f"\n{self.AGENT_NAME}: {response}")
        except KeyboardInterrupt:
            print(f"\n\n{self.AGENT_NAME}: Session interrupted. Goodbye.")
