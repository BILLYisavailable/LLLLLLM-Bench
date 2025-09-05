# Needle in Haystack Evaluation

A simplified Needle in Haystack testing tool, supporting both single-needle and multi-needle tests.

## Usage

```bash
# Run all tests
python run_needle_tests.py --model ../models/Llama-3.1-8B-Instruct

# Run only the single-needle test
python run_needle_tests.py --test-type single

# Run only the multi-needle test
python run_needle_tests.py --test-type multi
