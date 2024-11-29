import argparse

# Initialize parser
parser = argparse.ArgumentParser(description="A program that processes arguments.")

# Adding arguments
parser.add_argument("-n", "--name", type=str, help="Your name")
parser.add_argument("-a", "--ageee", type=int, help="Your age")
parser.add_argument("--verbose", action="store_true", help="Enable verbose mode")

# Parse arguments
args = parser.parse_args()

# Access the arguments
print(f"Name: {args.name}")
print(f"Age: {args.age}")
if args.verbose:
    print("Verbose mode is enabled!")
