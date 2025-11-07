# Using write_response.py

This script writes a specified text content to a file or standard output. It can take input directly from the command line, from a file, or from standard input.

## Usage

```bash
python write_response.py --which a --text "Hello, World!"

python write_response.py --which b --text "Line1\nLine2"
```

# Using run_single_pair.py

This script compares the outputs of two response files using different scoring methods.

## Usage

Run this from the root directory:

```bash
python .\tmp\run_single_pair.py
```

# Using nlp_research.py

This script performs various NLP research tasks based on the specified mode.

```bash
python nlp_research.py --test_case test_1_136060 --use-human-bias --gold 3

python nlp_research.py --test_case test_2_211333 --use-human-bias --answer-weight 0 --human-weight 0.6 --style-ref-file .\tmp\test_2_211333\test_a.txt

python nlp_research.py --test_case test_2_211333 --use-human-bias --answer-weight 0 --human-weight 0.5 --enable-concise-sweetspot --concise-target 150 --concise-lower 90 --concise-upper 350 --concise-min-score 0.5

python nlp_research.py --test_case test_2_211333 --use-human-bias --answer-weight 0 --human-weight 0.55 --enable-concise-sweetspot --concise-target 160 --concise-lower 70 --concise-upper 400 --concise-min-score 0.45 --enable-verbosity-penalty --verbosity-max-breaks 6 --verbosity-slope 0.02 --verbosity-max-penalty 0.15 --verbosity-weight 0.3

python nlp_research.py --test_case test_3_1233961 --use-human-bias --gold 3
```
