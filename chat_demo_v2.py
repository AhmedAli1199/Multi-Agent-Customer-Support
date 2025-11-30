"""
Interactive Chat Demo for Enhanced Multi-Agent System V2
Chat with TechGear Electronics customer support system.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from orchestration.graph_v2 import multi_agent_workflow_v2
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)


def print_banner():
    """Print welcome banner"""
    print("\n" + "=" * 80)
    print(Fore.CYAN + Style.BRIGHT + "🏪 TECHGEAR ELECTRONICS - CUSTOMER SUPPORT CHAT".center(80))
    print(Fore.CYAN + "Enhanced Multi-Agent System V2".center(80))
    print("=" * 80)
    print(f"\n{Fore.WHITE}Welcome to TechGear Electronics!{Style.RESET_ALL}")
    print("Your trusted technology partner since 2015.\n")
    print(f"{Fore.GREEN}I can help you with:{Style.RESET_ALL}")
    print("  • 📱 Product search and recommendations")
    print("  • 📦 Order management (cancel, track, modify)")
    print("  • 💰 Refunds and returns")
    print("  • 📋 Company policies and information")
    print("  • 🎧 Product comparisons and specs")
    print(f"\n{Fore.YELLOW}Try asking:{Style.RESET_ALL}")
    print('  - "Show me your laptops"')
    print('  - "Cancel order 12345"')
    print('  - "What\'s your return policy?"')
    print('  - "I need wireless headphones for the gym"')
    print(f"\n{Fore.CYAN}Type 'quit' or 'exit' to end the conversation.{Style.RESET_ALL}\n")
    print("=" * 80 + "\n")


def chat():
    """Run interactive chat session"""
    print_banner()

    conversation_history = []

    while True:
        # Get user input
        try:
            user_input = input(f"{Fore.YELLOW}You: {Style.RESET_ALL}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{Fore.CYAN}Thank you for chatting with TechGear Electronics! Have a great day!{Style.RESET_ALL}\n")
            break

        # Check for exit
        if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
            print(f"\n{Fore.CYAN}Thank you for chatting with TechGear Electronics! Have a great day!{Style.RESET_ALL}\n")
            break

        if not user_input:
            continue

        # Add to conversation history
        conversation_history.append({
            "role": "user",
            "content": user_input
        })

        # Initialize state
        state = {
            "customer_query": user_input,
            "conversation_history": conversation_history[:-1],  # Exclude current message
            "agent_sequence": [],
            "resolution_status": "pending"
        }

        print(f"\n{Fore.CYAN}TechGear Support is thinking...{Style.RESET_ALL}\n")

        try:
            # Run workflow
            result = multi_agent_workflow_v2.invoke(state)

            # Get response
            response = result.get("final_response", "I apologize, but I'm having trouble processing your request.")

            # Print response
            print(f"{Fore.GREEN}TechGear Support: {Style.RESET_ALL}{response}\n")

            # Show agent path (for demo purposes)
            agent_sequence = result.get("agent_sequence", [])
            if agent_sequence:
                print(f"{Fore.MAGENTA}[Agent Path: {' → '.join(agent_sequence)}]{Style.RESET_ALL}\n")

            # Add to conversation history
            conversation_history.append({
                "role": "assistant",
                "content": response
            })

        except Exception as e:
            print(f"{Fore.RED}❌ Error: {Style.RESET_ALL}{str(e)}\n")
            print(f"{Fore.YELLOW}Please try rephrasing your question or contact support at 1-800-TECHGEAR{Style.RESET_ALL}\n")


if __name__ == "__main__":
    chat()
