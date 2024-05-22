from __init__ import *

def metafunction(c, k):
    """
    Return a code that contains k instances of the the same program, yet with different
     arrengements of the c internal loops.
    """
    assert c > 0 and isinstance(c, int)
    assert k > 0 and isinstance(k, int)    
    
    definition = 'def f(n):\n'
    definition += '\t' + f"n_0={random.choice([-1, 0, 1])}"
    for i in range(1,c):
        definition += f"; n_{i}={random.choice([-1, 0, 1])}"
    definition += '\n'
    choices = ['*=-1', '*=2', '*=-2', '+=1', '+=-1', '+=2', '-=2']  # if we use random.choice we mess up with the seed and get very similar loops :(
    forloop = lambda: '\t' + f"for _ in range(n):\n"
    body_gen = lambda i: '\t'*2 + f"n_{i}" + random.choice(choices) + '\n'
    loops = [forloop() + body_gen(i) for i in range(c)]

    permutational_loops = []
    for _ in range(k):
        random.shuffle(loops)
        permutational_loops.append(cp.deepcopy(loops))
	
    ret = f"\treturn sum({[f'n_{i}' for i in range(c)]})"
    programs = []
    for loop in permutational_loops:            
        s = ''
        for l in loop:
            s += l
        programs.append(definition + s + ret.replace("'", "") + '\n\n')
    
    return ''.join(programs)
    
if __name__ == "__main__":
    """
    Example usage
    python3 redundant_computation.py --model gpt35 --query-method cot --num-programs 2 --loops 3 --instructions 3 --redundant-programs 3 --verbose true --sleep 5
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
    parser.add_argument("-l", "--loops", dest="n_loops", type=int, default=1,
                        help="Number of sequential loops generated.")
    parser.add_argument("-i", "--instructions", dest="instructions_perloop", type=int, default=1,
                        help="Number of instructions executed in each loop.")
    parser.add_argument("-r", "--redundant-programs", dest="redundant_programs", type=int, default=3,
                        help="Number of redundant programs generated.")
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
    n_loops = int(args.n_loops)
    instructions_perloop = int(args.instructions_perloop)
    redundant_programs = int(args.redundant_programs)
    verbose = (True if args.verbose.lower()=="true" else False)
    sleep_time = int(args.sleep_time)

    # "Global" vars TODO: add argparse support
    task = 'complexity-redundant'
    file_query = './prompts/nested-loops-redundant.txt'
    # K-shot params
    file_kshot = './prompts/kshot/nested-loops-redundant.txt'
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
    programs = [metafunction(n_loops, redundant_programs) for _ in range(n_programs)]

    # Ground truth label for each program
    ground_truth = []
    for p in programs:
        exec(p)
        ground_truth.append(f(instructions_perloop))  # f is the name of the function that encloses each loop

    assert len(programs) == len(ground_truth)

    # Logs prefix
    os.makedirs(logs_folder, exist_ok=True)
    logs_path = f'{logs_folder}{task}-{query_method}.txt'
    with open(logs_path, 'a+') as file_:
        file_.write('#'*30 + '\n')
        file_.write(str(datetime.datetime.now()) + '\n')
        file_.write(f"n_programs: {n_programs}, n_loops: {n_loops}, instructions_perloop: {instructions_perloop}, redundant_programs={redundant_programs}\n")
        
    query = read_prompts(file_query, query_method)  # Get the query
    predictions, lenghts = [], []
    correct, total = 0, 0
    for code,y in zip(programs, ground_truth):
        # Substitute the tags in each prompt
        prompt = cp.deepcopy(query)
        prompt = prompt.replace('@code@', str(code))
        prompt = prompt.replace('@input@', str(instructions_perloop))
        prompt = prompt.replace('@nfunc@', str(redundant_programs))   
             
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
                while len(r_multiple) < self_num_queries:
                    try:
                        r_multiple.append(queryLLM(prompt, config_file, model_name, self_temperature))
                    except:
                        pass
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
                
        # Resuts
        if verbose:
            print(f"\n<prompt>\n{prompt}\n</prompt>\n")
            print(f"<response>\n{response}\n</response>\n")
            print(f"<ground-truth>\n{y}\n</ground-truth>\n")

        predictions.append([]); lenghts.append([])
        predictions[-1].append(response)
        lenghts[-1].append(len(prompt))
        
        try:
            r = re.findall(r"<result>(.*)</result>", response)
            y_hat = list(r[-1])
            correct += (1 if y_hat == list(r) else 0)
        except:
            y_hat = "None"
        total += 1

        # Logs
        with open(logs_path, 'a+') as file_:
            file_.write(f"\n<prompt>\n{prompt}\n</prompt>\n")
            file_.write(f"<response>\n{response}\n</response>\n")
            file_.write(f"<ground-truth>\n{y}\n</ground-truth>\n")

        sleep(sleep_time)  # sometimes the server crashes, so let's give it a break :) TODO: sync calls

    with open(logs_path, 'a+') as file_:
        file_.write(f"\n<accuracy>\n{correct/total}\n</accuracy>\n")