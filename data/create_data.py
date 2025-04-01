from data_utilities import *
import argparse

def main(num_of_samples, input_file_path, output_file_path, create_metadata_flag,reverse,function_to_call,file_name):
    # Create datasets
    if reverse: 
        if not input_file_path:
            raise ValueError("Input file path must be provided for reverse examples generation.")
        
        if not file_name:
            raise ValueError("File name must be provided for reverse examples generation.")
        input_file_path2 = input_file_path + "/" + file_name
        file_name = "reverse_" + file_name

        input_file_path = input_file_path + file_name
        generate_reverse_examples(input_file_path2, input_file_path)
        print(f"Reverse examples created at {input_file_path}")
    else:
        if args.function_to_call == "make_addition_examples_all3digits":
            file_name = "3digit_add_examples_" + str(num_of_samples)
            input_file_path = input_file_path + "/" + file_name
            make_addition_examples_all3digits(
                num_samples=args.num_of_samples,
                input_file_path=args.input_file_path
            )
        elif args.function_to_call == "generate_partial_carry_pairs":
            file_name = "partial_carry_hundreds_" + str(num_of_samples)
            input_file_path = input_file_path + "/" + file_name
            generate_partial_carry_pairs(
                input_file_path=args.input_file_path,
                num_pairs=args.num_of_samples
            )
        elif args.function_to_call == "generate_addition_with_tens_hundreds_carry":
            file_name = "partial_carry_with_tens_hundreds_carry" + str(num_of_samples)
            input_file_path = input_file_path + "/" + file_name
            generate_addition_with_tens_hundreds_carry(
                input_file_path=args.input_file_path,
                num_pairs=args.num_of_samples
            )

    
    print(f"Datasets created with {num_of_samples} samples at {input_file_path}")
    # Create metadata if the flag is set to True
    if create_metadata_flag:
        output_file_path = input_file_path + ""
        data = ""
        with open(input_file_path, 'r') as f:
            data = f.read()
        create_meta_data(data, output_file_path)
        print(f"Metadata created for datasets at {output_file_path}")

if __name__ == "__main__":
    # Example usage
    # python create_data.py --num_of_samples 1000 --input_file_path "input.txt" --output_file_path "output_dataset" --create_metadata_flag --reverse
    parser = argparse.ArgumentParser(description="Create datasets and metadata.")
    parser.add_argument("--num_of_samples", type=int, required=True, help="Number of samples to generate")
    parser.add_argument("--input_file_path", type=str, required = False, help="Path to the input file")
    parser.add_argument("--output_file_path", type=str, required=True, help="Path to the output dataset file")
    parser.add_argument("--create_metadata_flag", action="store_true", help="Set to True to create metadata")
    parser.add_argument("--reverse", action="store_true", help="Set to True to generate reverse examples")
    parser.add_argument("--file_name", type=str, default="data_10000", help="Name of the dataset file to be created. Default is 'data_10000'.")
    parser.add_argument("--function_to_call", type=str, required=True, choices=[
        "make_addition_examples_all3digits",
        "generate_partial_carry_pairs",
        "generate_addition_with_tens_hundreds_carry"
    ], help="Specify the function to call")

    args = parser.parse_args()
    
    num_of_samples = args.num_of_samples
    input_file_path = args.input_file_path
    output_file_path = args.output_file_path
    create_metadata_flag = args.create_metadata_flag
    reverse1 = args.reverse
    function_to_call = args.function_to_call
    file_name = args.file_name


    main(num_of_samples, input_file_path, output_file_path, create_metadata_flag, reverse1,function_to_call, file_name)