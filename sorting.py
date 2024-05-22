from __init__ import *

if __name__ == "__main__":
    """
    Example usage
    python3 sorting.py --model gpt35 --query-method cot --algorithm-type iterative --num-vectors 2 --length 10 --verbose true --sleep 5
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", "--config", dest="config_file", type=str, default='config_multiple.json',
                        help="Config file.")
    parser.add_argument("-m", "--model", dest="model_name", type=str, default='gpt35',
                        help="Model to query. So far ['bard', 'gpt35', 'gpt4', 'command', 'claude2', 'jurassic'] are supported.")
    parser.add_argument("-q", "--query-method", dest="query_method", type=str, default='cot',
                        help="Query method. So far [cot, kshot-cot, code, kshot-code] are supported.")
    parser.add_argument("-temp", "--temperature", dest="temperature", type=float, default=0.,
                    help="Temperature. For open-source models, it must be strictly positive. This value is ignored if --query-method is self-consistency.")
    parser.add_argument("-t", "--algorithm-type", dest="algo_type", type=str, default='iterative',
                        help="Whether to prompt the iterative or recursive rotuine.")
    parser.add_argument("-n", "--num-vectors", dest="n_samples", type=int, default=10,
                        help="Number of vectors (to be sorted) queried.")
    parser.add_argument("-l", "--length", dest="v_length", type=int, default=10,
                        help="Length of each vector.")
    parser.add_argument("-verbose", "--verbose", dest="verbose", type=str, default="True",
                    help="Verbose mod (console).")
    parser.add_argument("-s", "--sleep", dest="sleep_time", type=int, default=5,
                        help="Sleep time in seconds between each request.")
    args = parser.parse_args()
    config_file = str(args.config_file)
    model_name = str(args.model_name)
    query_method = str(args.query_method)
    temperature = float(args.temperature)
    algo_type = str(args.algo_type).lower()
    n_samples = int(args.n_samples)
    v_length = int(args.v_length)
    verbose = (True if args.verbose.lower()=="true" else False)
    sleep_time = int(args.sleep_time)

    if algo_type == 'iterative':
        column_algo = 'Iterative Code'
    elif algo_type == 'recursive':
        column_algo = 'Recursive Code'
    else:
        raise Exception(f"{algo_type} is not a recognized method.")

    # "Global" vars TODO: add argparse support
    task = 'sorting'
    file_query = './prompts/sorting.txt'
    # K-shot params
    file_kshot = './prompts/kshot/sorting.txt'
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

    # Extract the sorting algorithms
    data = pd.read_csv('./code/sorting_algorithms-cosm.csv')
    data = data.iloc[:]

    # Generate the test cases
    unsorted_vectors = np.random.randint(0, 100, (n_samples, v_length)).tolist()  # generate a random unsorted array
    ground_truth = cp.deepcopy(unsorted_vectors)
    for el in ground_truth:
        el.sort()

    assert len(unsorted_vectors) == len(ground_truth)

    # Logs prefix
    os.makedirs(logs_folder, exist_ok=True)
    logs_path = f'{logs_folder}{task}-{query_method}.txt'
    with open(logs_path, 'a+') as file_:
        file_.write('#'*30 + '\n')
        file_.write(str(datetime.datetime.now()) + '\n')
        file_.write(f"algo_type: {algo_type}, n_samples: {n_samples}, v_length: {v_length}\n")

    for index, row in data.iterrows():
        query = read_prompts(file_query, query_method)  # Get the query
        correct, total = 0, 0
        with open(logs_path, 'a+') as file_:
            file_.write(f"algorithm: {row['Algorithm']}\n")
            
        for v,y in zip(unsorted_vectors, ground_truth):
            # Substitute the tags in each prompt
            prompt = cp.deepcopy(query)
            prompt = prompt.replace('@code@', str(row[column_algo]))
            prompt = prompt.replace('@input@', str(v))
            
            # Substitute the kshot tags
            if f"@{kshot_tag}@" in prompt:
                illustration = read_prompts(file_kshot, kshot_tag)
                prompt = prompt.replace(f"@{kshot_tag}@", illustration)
            
            # Substitute other tags (tot, etc.)
            prompt = prompt.replace(tot_num_splits_tag, str(tot_num_splits))
            prompt = prompt.replace(tot_num_experts_tag, str(tot_num_experts))
        
        try:
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

            try:
                if f"{str(y)}".replace(" ", "") in response.replace(" ", ""):
                    correct += 1
            except:
                response = "None"
                print(f"Something went wrong with the automatic evaluation of performances, please inspect the logs manually.")
            total += 1
                
            # Resuts
            if verbose:
                print(f"\n<prompt>\n{prompt}\n</prompt>\n")
                print(f"<response>\n{response}\n</response>\n")
                print(f"<ground-truth>\n{y}\n</ground-truth>\n")

            # Logs
            with open(logs_path, 'a+') as file_:
                file_.write(f"\n<prompt>\n{prompt}\n</prompt>\n")
                file_.write(f"<response>\n{response}\n</response>\n")
                file_.write(f"<ground-truth>\n{y}\n</ground-truth>\n")
            
            sleep(sleep_time)
            
        except:
            sleep(int(2*sleep_time))  # sometimes the server crashes, so let's give it a break :) TODO: sync calls
        
        with open(logs_path, 'a+') as file_:
                file_.write(f"\n<Accuracy> Alg {row['Algorithm']}-{algo_type} ({total} samples): {correct/total}</accuracy>\n\n")

