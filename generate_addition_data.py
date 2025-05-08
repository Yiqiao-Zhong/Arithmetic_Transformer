import random
import argparse
from pathlib import Path

def generate_addition_problem(num_operands):
    """Generate an addition problem with specified number of operands where result is 2 digits."""
    while True:
        # Generate n+1 numbers (n operands plus one more to ensure we can get a 2-digit result)
        numbers = [random.randint(1, 9) for _ in range(num_operands + 1)]
        result = sum(numbers)
        if 10 <= result <= 99:  # Ensure 2-digit result
            # Format the numbers with + between them
            problem = "+".join(map(str, numbers))
            return f"{problem}={result}\n"

def generate_data(num_operands, num_samples, output_path):
    """Generate addition data and write to file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for _ in range(num_samples):
            line = generate_addition_problem(num_operands)
            f.write(line)

def reverse_results(input_path, output_path):
    """Read the input file, reverse the results, and write to output file."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            # Split the line into problem and result
            problem, result = line.strip().split('=')
            # Reverse the result
            reversed_result = result[::-1]
            # Write the new line with reversed result
            f_out.write(f"{problem}={reversed_result}\n")

def main():
    parser = argparse.ArgumentParser(description='Generate addition data or reverse existing data')
    parser.add_argument('--mode', choices=['generate', 'reverse'], required=True,
                      help='Mode: generate new data or reverse existing data')
    parser.add_argument('--operands', type=int, choices=range(1, 9),
                      help='Number of operands (1 to 8) - required for generate mode')
    parser.add_argument('--samples', type=int,
                      help='Number of samples to generate - required for generate mode')
    parser.add_argument('--input', type=str,
                      help='Input file path - required for reverse mode')
    parser.add_argument('--output', type=str, required=True,
                      help='Output file path')
    
    args = parser.parse_args()
    
    if args.mode == 'generate':
        if args.operands is None or args.samples is None:
            parser.error("--operands and --samples are required in generate mode")
        generate_data(args.operands, args.samples, args.output)
    else:  # reverse mode
        if args.input is None:
            parser.error("--input is required in reverse mode")
        reverse_results(args.input, args.output)

if __name__ == '__main__':
    main() 