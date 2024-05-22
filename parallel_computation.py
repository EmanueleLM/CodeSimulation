from __init__ import *

def metafunction(n, m, vars_per_program=3, mode='p&c'):
    """
    This function creates a program of length n and with v variables connected by simple operations.
    
    n is the number of operations returned
    m is the number of independent critical paths. n must be divisible by m
    vars_per_program is the number of variables per program. Must be at least 3
    mode is either p&c (parallelizable and continuous), p (parallelizable), or None (a program of max length with a unique critical path will be created)

    Examples:
    # Program with n=20 instructions and m=1 critical path
    p = metafunction(20, 1, 3, None)
    print(p)

    # Program with n=20 instructions, m=4 continuous critical path programs of length 20/4
    p = metafunction(20, 5, 3, 'p')
    print(p)

    # Program with n=20 instructions, m=4 non-continuous critical path programs of length 20/4
    p = metafunction(20, 5, 3, 'p&c')
    print(p)
    """
    assert n >= 0 and isinstance(n, int)
    assert n % m == 0 and isinstance(m, int)
    assert vars_per_program >= 3 and isinstance(vars_per_program, int)
    assert ((m == 1) and (mode is None)) or ((m > 1) and (mode is not None))

    v = vars_per_program * m
    ops = ['@1@ = @2@', '@1@ += @2@', '@1@ -= @2@']
    variables = [f'a{i}' for i in range(v)]

    if mode is None:
        start_inj = [0]
        length = n
    elif mode in ['p', 'p&c']:
        start_inj = [i for i in range(0, n, int(n/m))]
        length = int(n/m)
    else:
        raise Exception(f"{mode} is not a supported mode.")

    program = ''
    program = '; '.join([f'a{i}={random.randint(-10, 10)}' for i in range(v)]) + '\n'
    p_and_c_program = [program]  # use a list to store each sub-program and then ricombine it
    if mode is None:
        for i in start_inj:
            for j in range(i, length, 1):
                op = random.choice(ops)
                line = cp.deepcopy(op)
                if j == 0:
                    src, dst = random.choice(variables), random.choice(variables)
                else:
                    if prev_ref == 0:  # previous was src, so need to set dst = prev_v
                        src, dst = random.choice(variables), prev_v

                    else:  # previous was dst, so we can set either set src or dst = prev_v
                        p = random.random()
                        if p > 0.5:  # prev_v goes to src
                            src, dst = prev_v, random.choice(variables)
                        else:
                            src, dst = random.choice(variables), prev_v

                p = random.random()
                if p > 0.5:  # prev_v goes to dst
                    prev_v = dst
                    prev_ref = 1
                else:  # prev_v goes to src
                    prev_v = src
                    prev_ref = 0

                line = line.replace('@1@', dst).replace('@2@', src)
                program += line + '\n'
                # print(line)
                # print(prev_ref, ('src' if prev_ref==0 else 'dst'))

    else:  # mode in ['p', 'p&c']:
        for ctr,i in enumerate(start_inj):
            vars = variables[ctr*vars_per_program:(ctr*vars_per_program)+vars_per_program]
            for j in range(0, length, 1):
                op = random.choice(ops)
                line = cp.deepcopy(op)                
                if j == 0:
                    src, dst = random.choice(vars), random.choice(vars)
                    prev_v = src
                else:
                    p = random.random()
                    if p > 0.5:
                        src, dst = prev_v, random.choice(vars)
                        prev_v = dst
                    else:
                        src, dst = random.choice(vars), prev_v
                        prev_v = src
                line = line.replace('@1@', src).replace('@2@', dst)
                program += line + '\n'
                p_and_c_program.append(line)

    # Re-order 'p&c'
    if mode == 'p&c':
        program = p_and_c_program[0]
        for i in range(0, length, 1):
            for j in start_inj:
                program += p_and_c_program[1+j+i] + '\n' # skip the vars declaration

    eval_function = f'sum([' + ','.join([f'a{i}' for i in range(v)]) + '])'
    
    return program, eval_function

if __name__ == "__main__":
    """
    Example usage
    python3 parallel_computation.py --model gpt35 --query-method cot --num-programs 2 --program-type p&c --num-vars 3 --program-len 10 --critical-path 5 --verbose true
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", "--config", dest="config_file", type=str, default='config_multiple.json',
                        help="Config file.")
    parser.add_argument("-m", "--model", dest="model_name", type=str, default='gpt35',
                        help="Model to query. So far ['bard', 'gpt35', 'gpt4', 'command', 'claude2', 'jurassic'] are supported.")
    parser.add_argument("-q", "--query-method", dest="query_method", type=str, default='cot',
                        help="Query method. So far [cot, kshot-cot, code, kshot-code] are supported.")
    parser.add_argument("-t", "--temperature", dest="temperature", type=float, default=0.,
                help="Temperature. For open-source models, it must be strictly positive. This value is ignored if --query-method is self-consistency.")
    parser.add_argument("-n", "--num-programs", dest="n_programs", type=int, default=10,
                        help="Number of programs queried.")
    parser.add_argument("-v", "--num-vars", dest="num_vars", type=int, default=5,
                        help="Number of variables per-program.")
    parser.add_argument("-l", "--program-len", dest="program_length", type=int, default=10,
                        help="Number of instructions in each program.")
    parser.add_argument("-p", "--program-type", dest="program_type", type=str, default='p',
                        help="Either p&c (parallelizable and continuous), p (parallelizable), or None (a program of \
                            max length with a unique critical path will be created).")
    parser.add_argument("-c", "--critical-path", dest="critical_path", type=int, default=5,
                        help="Length of the critical path in each program.")
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
    critical_path = int(args.critical_path)
    program_type = str(args.program_type).lower()
    verbose = (True if args.verbose.lower()=="true" else False)
    sleep_time = int(args.sleep_time)

    if program_type in ['p&c', 'p']:
        pass 
    elif program_type == 'none':
        program_type = None
    else:
        raise Exception(f"{program_type} is not a valid value for program_type.")

    assert critical_path <= program_length

    # "Global" vars TODO: add argparse support
    task = 'code-exec-parallel'  # used only to name log files
    file_query = './prompts/code-exec-parallel.txt'
    # K-shot params
    file_kshot = './prompts/kshot/code-exec-parallel.txt'
    kshot_tag = 'illustration'
    # Self-consistency params
    self_num_queries = 3
    self_temperature = 0.1
    # ToT params
    tot_num_splits = 3
    tot_num_experts = 3
    tot_num_splits_tag = "@num_chunks@"
    tot_num_experts_tag = "@num_experts@"
    # Other vars
    logs_folder = f'./logs/{model_name}/'

    # Create some programs
    programs = []
    for i in range(n_programs):
        programs.append({})
        programs[i]['ops'] = program_length
        programs[i]['vars'] = num_vars
        programs[i]['code'], programs[i]['eval'] = metafunction(n=program_length, m=critical_path, vars_per_program=num_vars, mode=program_type)

    # Ground truth label for each program
    ground_truth = []
    for pp in programs:
        exec(pp['code'])
        r = eval(pp['eval'])
        ground_truth.append(r)

    assert len(programs) == len(ground_truth)

    # Logs prefix
    os.makedirs(logs_folder, exist_ok=True)
    logs_path = f'{logs_folder}{task}-{query_method}.txt'
    with open(logs_path, 'a+') as file_:
        file_.write('#'*30 + '\n')
        file_.write(str(datetime.datetime.now()) + '\n')
        file_.write(f"n_programs: {n_programs}, program_length: {program_length}, num_vars: {num_vars}, critical_path: {critical_path}, program_type: {program_type}\n")        
        
    query = read_prompts(file_query, query_method)  # Get the query
    correct, total = 0, 0
    predictions, lenghts = [], []
    for p,y in zip(programs, ground_truth):
        code = p['code'] 
        var_to_predict = p['vars']-1  # var to predict (last one for critical path (by construction))
        # Substitute the tags in each prompt
        prompt = cp.deepcopy(query)
        prompt = prompt.replace('@code@', str(code))
        prompt = prompt.replace('@v@', str(var_to_predict))

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
        
        predictions.append([]); lenghts.append([])
        predictions[-1].append(response)
        lenghts[-1].append(len(prompt))
        
        try:
            r = re.findall(r"<result>(.*)</result>", response)
            y_hat = int(r[-1])
            correct += (1 if y_hat == int(y) else 0)
        except:
            y_hat = "None"
        total += 1
        
        # Resuts
        if verbose:
            print(f"\n<prompt>\n{prompt}\n</prompt>\n")
            print(f"<response>\n{response}\n</response>\n")
            print(f"<ground-truth>\na{var_to_predict}={y}\n</ground-truth>\n")

        # Logs
        with open(logs_path, 'a+') as file_:
            file_.write(f"\n<prompt>\n{prompt}\n</prompt>\n")
            file_.write(f"<response>\n{response}\n</response>\n")
            file_.write(f"<ground-truth>\n{y}\n</ground-truth>\n")

        sleep(sleep_time)  # sometimes the server crashes, so let's give it a break :) TODO: sync calls
    
    with open(logs_path, 'a+') as file_:
        file_.write(f"\n<accuracy>\n{correct/total}\n</accuracy>\n")