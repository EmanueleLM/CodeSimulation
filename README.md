# Intro
This repo allows to replicate all the experiments for the paper 'Code Simulation Chiallenges for Large Language Models'.
You can read the article here: https://arxiv.org/pdf/2401.09074

And cite it with the following bibtex:
```
@misc{lamalfa2024code,
      title={Code Simulation Challenges for Large Language Models}, 
      author={Emanuele La Malfa and Christoph Weinhuber and Orazio Torre and Fangru Lin and Samuele Marro and Anthony Cohn and Nigel Shadbolt and Michael Wooldridge},
      year={2024},
      eprint={2401.09074},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

For any further inquiries, please open an issue on the repo or send me an email at amailaddress0x90@gmail.com

# 1. Install dependencies
First thing first, you need to install all the dependencies. 
We encourage creating a virtual environment (venv works fine).
Then run:
```
pip install -r requirements.txt
```

# 2. Insert your key
Populate `config_multiple.json` with the appropriate keys for each Language Models you want to use.
For example, add the org and key for GPT-3.5
```
    "gpt35": {
        "organization": "YOUR ORG", --> your org
        "name":"gpt-3.5-turbo",
        "key":"YOUR KEY" --> your LLM key
    }
```

# 3. Running experiments for straight-line code simulation
The instructions to run the experiments for all the benchmarks are reported below.
For simplicity, each benchmark is encapsulated in a standalone python file.
For more information on the arguments to pass to the command line, please run filename.py --help

1. Straight-line code simulation
```
    python3 codeexec.py --model gpt35 --query-method cot --num-programs 2 --num-vars 5 --program-len 10 --verbose true
```

2. Critical path
```
    python3 critical_path.py --model gpt35 --query-method cot --num-programs 2 --num-vars 5 --program-len 10 --critical-path 5 --verbose true
```

3. Approximate computation
```
    python3 approximate_computation.py --model gpt35 --query-method cot --num-programs 2 --loops 3 --instructions 3 --verbose true --sleep 5
```

4. Reduntant computation
```
    python3 redundant_computation.py --model gpt35 --query-method cot --num-programs 2 --loops 3 --instructions 3 --redundant-programs 3 --verbose true --sleep 5
```

5. Nested loops
```
    python3 redundant_computation.py --model gpt35 --query-method cot --num-programs 2 --loops 3 --instructions 3 --redundant-programs 3 --verbose true --sleep 5

```

6. Sorting algorithms
```
    python3 sorting.py --model gpt35 --query-method cot --algorithm-type iterative --num-vectors 2 --length 10 --verbose true --sleep 5
```

7. Generic problems (with CoT and CoSm)
```
    python3 cosm_generic_problem.py --model gpt35 --query-method cot --num-experiments 1 --verbose true --sleep 5 --algorithm fibo
```
