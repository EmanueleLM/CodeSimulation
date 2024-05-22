from __init__ import *

def metafunction(n, v, filter_ops='any'):
    """
    n is the number of operations returned
    v is the number of variables declared in the code
    filter_ops is the type of operations allowed
    """
    assert n > 0 and isinstance(n, int)

    if filter_ops == 'any':
        ops = ['@1@ = @2@', '@1@ += @2@', '@1@ -= @2@', '@1@ |= @2@', '@1@ &= @2@']
    elif filter_ops == 'sum':
        ops = ['@1@ += @2@', '@1@ -= @2@']
    elif filter_ops == 'mov':
        ops = ['@1@ = @2@']
    elif filter_ops == 'and':
        ops = ['@1@ |= @2@', '@1@ &= @2@']
    elif filter_ops == 'nological':
        ops = ['@1@ = @2@', '@1@ += @2@', '@1@ -= @2@']
    else:
        raise Exception(f'{filter_ops} is not a supported value for filter_ops.')

    variables = [f'a{i}' for i in range(v)]
    program = ''
    program = '; '.join([f'a{i}={random.randint(-10, 10)}' for i in range(v)]) + '\n'
    for _ in range(n):
        op = random.choice(ops)
        line = cp.deepcopy(op)
        src = random.choice(variables)
        if '-' in op:  # avoid operation ai -= ai 
            set_dst = (set(variables) if isinstance(variables, list) else set([variables]))
            dst = random.choice(list(set_dst - set([src])))
        else:
            dst = random.choice(variables)
        line = line.replace('@1@', dst).replace('@2@', src)
        program += line + '\n'
    
    return program

if __name__ == "__main__":
    """
    Example usage
    python3 codeexec.py --model gpt35 --query-method cot --num-programs 2 --num-vars 5 --program-len 10 --verbose true
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", "--config", dest="config_file", type=str, default='config_multiple.json',
                        help="Config file.")
    parser.add_argument("-m", "--model", dest="model_name", type=str, default='gpt35',
                        help="Model to query. So far ['bard', 'gpt35', 'gpt4', 'command', 'claude2', 'jurassic', 'llama', 'llamacode', 'llama3'] are supported.")
    parser.add_argument("-q", "--query-method", dest="query_method", type=str, default='cot',
                        help="Query method. So far [cot, cot-linebyline, kshot-cot, self-consistency, code-self-consistency, tree-of-thoughts, code, kshot-code] are supported.")
    parser.add_argument("-t", "--temperature", dest="temperature", type=float, default=0.,
                    help="Temperature. For open-source models, it must be strictly positive. This value is ignored if --query-method is self-consistency.")
    parser.add_argument("-n", "--num-programs", dest="n_programs", type=int, default=10,
                        help="Number of programs queried.")
    parser.add_argument("-v", "--num-vars", dest="num_vars", type=int, default=5,
                        help="Number of variables per-program.")
    parser.add_argument("-l", "--program-len", dest="program_length", type=int, default=10,
                        help="Number of instructions in each program.")
    parser.add_argument("-f", "--filter-ops", dest="filter_ops", type=str, default='any',
                    help="Type of instructions in each program. So far ['any', 'nological', 'sum', 'mov', 'and']")
    parser.add_argument("-verbose", "--verbose", dest="verbose", type=str, default="True",
                    help="Verbose mod (console).")
    parser.add_argument("-s", "--sleep", dest="sleep_time", type=int, default=5,
                        help="Sleep time in seconds between each request.")
    args = parser.parse_args()
    config_file = str(args.config_file)
    model_name = str(args.model_name)
    query_method = str(args.query_method)
    temperature = float(args.temperature)
    n_programs = int(args.n_programs)
    num_vars = int(args.num_vars)
    program_length = int(args.program_length)
    filter_ops = str(args.filter_ops)
    verbose = (True if args.verbose.lower()=="true" else False)
    sleep_time = int(args.sleep_time)
     
    # "Global" vars TODO: add argparse support
    task = 'code-exec'
    file_query = './prompts/code-exec.txt'
    # K-shot params
    file_kshot = './prompts/kshot/code-exec.txt'
    kshot_tag = 'illustration'
    # Self-consistency params
    self_num_queries = 3
    self_temperature = 0.1
    # ToT params
    tot_num_splits = 3
    tot_num_experts = 3
    tot_num_splits_tag = "@tot-chunks@"
    tot_num_experts_tag = "@tot-experts@"
    # Other vars
    logs_folder = f'./logs/{model_name}/'
    
    # Create some programs
    programs = []
    for i in range(n_programs):
        programs.append({})
        programs[i]['ops'] = program_length
        programs[i]['vars'] = num_vars
        programs[i]['code'] = metafunction(program_length, num_vars, filter_ops)

    # Ground truth label for each program
    ground_truth = []
    for pp in programs:
        exec(pp['code'])
        ground_truth.append([eval(f'a{v}') for v in range(pp['vars'])])

    assert len(programs) == len(ground_truth)

    # Logs prefix
    os.makedirs(logs_folder, exist_ok=True)
    logs_path = f'{logs_folder}{task}-{query_method}-{filter_ops}.txt'
    with open(logs_path, 'a+') as file_:
        file_.write('#'*30 + '\n')
        file_.write(str(datetime.datetime.now()) + '\n')
        file_.write(f"n_programs: {n_programs}, program_length: {program_length}, num_vars: {num_vars}\n")        
        
    query = read_prompts(file_query, query_method)  # Get the query
    predictions, lenghts = [], []
    correct, total = 0, 0
    for p,y in zip(programs, ground_truth):
        code = p['code'] 
        var_to_predict = random.randint(0, p['vars']-1)  # random var to predict
        
        # Substitute the tags in each prompt
        prompt = cp.deepcopy(query)
        prompt = prompt.replace('@v@', str(var_to_predict))
        # For the linebyline we only substitute the first line
        if "linebyline" not in query_method:
            prompt = prompt.replace('@code@', str(code))
        else:
            code = code.split('\n')
            prompt = prompt.replace('@code@', str(code[0]))
            
        # Substitute the kshot tags
        if f"@{kshot_tag}@" in prompt:
            illustration = read_prompts(file_kshot, kshot_tag)
            prompt = prompt.replace(f"@{kshot_tag}@", illustration)
            
        # Substitute other tags (tot, etc.)
        prompt = prompt.replace(tot_num_splits_tag, str(tot_num_splits))
        prompt = prompt.replace(tot_num_experts_tag, str(tot_num_experts))
        
        if "linebyline" not in query_method:
            # Self-consistency requires multiple answers and majorioty vote
            if query_method == "self-consistency":
                r_multiple = []
                for _ in range(self_num_queries):
                    r_multiple.append(queryLLM(prompt, config_file, model_name, self_temperature))
                    sleep(sleep_time)
                r_multiple = [re.search(r'<result>(.*)</result>', r) for r in r_multiple]
                r_temp = [r.group(1) if r is not None else "None" for r in r_multiple]
                counter = list(collections.Counter(r_temp).values())
                # If there is no majority answer, take the temperature 0.
                if max(counter) > 1:
                    response = max(r_temp, key=r_temp.count)  # max frequency item
                else:
                    response = r_temp[0]  # take temperature 0.
                response = f"<result>{response}</result>"   
            else:
                response = queryLLM(prompt, config_file, model_name, temperature)  # prompt the LLM
        # Linebyline requires prompting the model with each line of code
        else:
            response = ""
            for instr in code:
                if len(instr) > 0:  # skip empty lines
                    prompt += f"\n{instr}"
                    r = queryLLM(prompt, config_file, model_name, temperature)
                    if r is not None:
                        response += r
                    prompt += f"\n{r}"
            prompt += "\nexit()\n"
            r = queryLLM(prompt, config_file, model_name, temperature)
            if r is not None:
                response = r

        # Resuts
        if verbose:
            print(f"\n<prompt>\n{prompt}\n</prompt>\n")
            print(f"<response>\n{response}\n</response>\n")
            print(f"<ground-truth>\na{var_to_predict}={y[var_to_predict]}\n</ground-truth>\n")

        # Append and parse result
        predictions.append([]); lenghts.append([])
        predictions[-1].append(response)
        lenghts[-1].append(len(prompt))
        
        try:
            r = re.findall(r"<result>(.*)</result>", response)
            y_hat = int(r[-1])
            correct += (1 if y_hat == int(y[var_to_predict]) else 0)
        except:
            y_hat = "None"
        total += 1

        # Write logs
        with open(logs_path, 'a+') as file_:
            file_.write(f"\n<prompt>\n{prompt}\n</prompt>\n")
            file_.write(f"<response>\n{response}\n</response>\n")
            file_.write(f"<ground-truth>\na{var_to_predict}={y[var_to_predict]}\n</ground-truth>\n")

        sleep(sleep_time)  # sometimes the server crashes, so let's give it a break :) TODO: sync calls

    with open(logs_path, 'a+') as file_:
        file_.write(f"\n<accuracy>\n{correct/total}\n</accuracy>\n")