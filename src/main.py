from src.interface.web_ui import WebUI
from src.interface.cli import CLI
import argparse

def main():
    parser = argparse.ArgumentParser(description="Advanced RAG Document QA System")
    parser.add_argument("--web", action="store_true", help="Launch web UI")
    parser.add_argument("--cli", action="store_true", help="Use command-line interface")
    args = parser.parse_args()
    
    if args.web:
        WebUI().launch()
    elif args.cli:
        CLI().run()
    else:
        print("Please specify an interface: --web or --cli")

if __name__ == "__main__":
    main()