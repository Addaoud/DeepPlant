from pathlib import Path

# Get the absolute path of the repository root
repo_root = Path(__file__).parent.resolve()
env_file = repo_root / ".env"

print("Setting up your local environment...")

# Read existing content if the file exists
content = ""
if env_file.exists():
    content = env_file.read_text()
else:
    print("Creating new .env file...")

# Append the path if it isn't already in the file
if "DEEPPLANTPATH=" not in content:
    with open(env_file, "a") as f:
        # Add a newline just in case the file doesn't end with one
        f.write(f'\nDEEPPLANTPATH="{repo_root}"\n')
    print(f"Successfully set DEEPPLANTPATH to: {repo_root}")
else:
    print("DEEPPLANTPATH is already set in .env. Skipping.")
