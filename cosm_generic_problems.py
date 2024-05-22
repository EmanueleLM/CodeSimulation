from __init__ import *

def get_problems(algorithm, n_samples):
    """
    Create a problem and return the ground truth for n_sample problems
    """
    problems = {'fibo':fibo, 'sort':sort, 'gauss':gauss, 'collatz':collatz, \
                'prime':is_prime, 'multiply':multiply, 'pi':pi_digit}
    return problems[algorithm](n_samples)

def fibo(n_samples):
    """
    Return fibo and padovan functions, and randomly sampled pairs of input-ouput. 
    """
    def f(n):
        a, b = 0, 1
        if n <=1:
            return n       
        else:
            for i in range(1, n):
                c = a + b
                a = b
                b = c
            return b
    
    def g(n):
        a, b = 1, 1
        c, d = 1, 1
        for i in range(3, n+1):
            d = a + b
            a = b
            b = c
            c = d 
        return d

    y_vanilla, y_variation = [], []
    samples = [i for i in range(n_samples)]
    for s in samples:
        y_vanilla.append(f(s))
        y_variation.append(g(s))
    return [inspect.getsource(f), inspect.getsource(g)], samples, (y_vanilla, y_variation)

def sort(n_samples):
    """
    Return sorting functions, and randomly sampled pairs of input-ouput. 
    """

    def f(v):
        n = len(v)
        for i in range(n):
            for j in range(0, n-i-1):
                if v[j] > v[j+1]:
                    v[j], v[j+1] = v[j+1], v[j]
        return v

    def g(v):
        n = len(v)
        for i in range(n):
            for j in range(0, n-i-1):
                if 0 > v[j] - v[j+1]:
                    v[j], v[j+1] = v[j+1], v[j]
        return v

    y_vanilla, y_variation = [], []
    samples = np.random.randint(0, 100, (n_samples, 10)).tolist()
    for s in samples:
        s_copy = cp.deepcopy(s)
        y_vanilla.append(f(s_copy))
        s_copy = cp.deepcopy(s)
        y_variation.append(g(s_copy))
    return [inspect.getsource(f), inspect.getsource(g)], samples, (y_vanilla, y_variation)

def gauss(n_samples):
    """
    Return gauss functions, and randomly sampled pairs of input-ouput. 
    """
    def f(n):
        tot = 0
        for i in range(n):
            tot += i
        return tot
    
    def g(n):
        tot = 0
        for i in range(n):
            tot += (i if i%2==0 else -i)
        return tot
    
    y_vanilla, y_variation = [], []
    samples = [i for i in range(n_samples)]
    for s in samples:
        y_vanilla.append(f(s))
        y_variation.append(g(s))
    return [inspect.getsource(f), inspect.getsource(g)], samples, (y_vanilla, y_variation)

def collatz(n_samples):
    """
    Return collatz functions, and randomly sampled pairs of input-ouput. 
    """
    def f(n):
        s = n
        while n != 1:
            if n % 2 == 0:
                n = n // 2
            else:
                n = 3 * n + 1
            s += n
        return s

    def g(n):
        s = n
        while n != 1:
            if n % 2 == 0:
                n = n // 2
                s += n
            else:
                n = 3 * n + 1
        return s

    y_vanilla, y_variation = [], []
    samples = [i for i in range(2, 2+n_samples)]
    for s in samples:
        y_vanilla.append(f(s))
        y_variation.append(g(s))
    return [inspect.getsource(f), inspect.getsource(g)], samples, (y_vanilla, y_variation)

def is_prime(n_samples):
    """
    Return True if a number is prime (vanilla)
    Return True if the successor of a number is prime (variation)
    """
    def f(n):
        if n < 2: return False
        for x in range(2, int(n**0.5) + 1):
            if n % x == 0:
                return False
        return True

    def g(n):
        n = n+1
        if n < 2: return False
        for x in range(2, int(n**0.5) + 1):
            if n % x == 0:
                return False
        return True
    
    y_vanilla, y_variation = [], []
    samples = [np.random.randint(1, 1000) for i in range(n_samples)]
    for s in samples:
        y_vanilla.append(f(s))
        y_variation.append(g(s))
    return [inspect.getsource(f), inspect.getsource(g)], samples, (y_vanilla, y_variation)

def multiply(n_samples):
    """
    Multiplication and its variation as iterative sum.
    """
    def f(a, b):
        return a*b

    def g(a, b):
        tot = 0
        for _ in range(b):
            tot += a
        return tot
    
    y_vanilla, y_variation = [], []
    samples = [(np.random.randint(2, 250), np.random.randint(2, 250)) for i in range(n_samples)]
    for a,b in samples:
        y_vanilla.append(f(a,b))
        y_variation.append(g(a,b))
    return [inspect.getsource(f), inspect.getsource(g)], samples, (y_vanilla, y_variation)

def pi_digit(n_samples):
    """
    n-th digit of pi (vanilla) and n+1 digit of pi (variation).
    """
    def f(n):
        from mpmath import mp
        mp.dps = n
        x = str(mp.pi)[n-1]
        return int(x)

    def g(n):
        from mpmath import mp
        mp.dps = n
        x = str(mp.pi)[n-1]
        return int(x)
    
    y_vanilla, y_variation = [], []
    samples = [np.random.randint(3, 250) for _ in range(n_samples)]
    for n in samples:
        y_vanilla.append(f(n))
        y_variation.append(g(n))
    return [inspect.getsource(f), inspect.getsource(g)], samples, (y_vanilla, y_variation)


if __name__ == "__main__":
    """
    Example usage
    python3 cosm_generic_problem.py --model gpt35 --query-method cot --num-experiments 1 --verbose true --sleep 5 --algorithm fibo
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
    parser.add_argument("-a", "--algorithm", dest="algorithm", type=str, default='fibo',
                        help="Algorithms and their variations. So far [fibo, sort, gauss, collatz, prime, multiply] are supported.")
    parser.add_argument("-n", "--num-experiments", dest="n_samples", type=int, default=10,
                        help="Number of examples to be queried.")
    parser.add_argument("-verbose", "--verbose", dest="verbose", type=str, default="True",
                    help="Verbose mod (console).")
    parser.add_argument("-s", "--sleep", dest="sleep_time", type=int, default=5,
                        help="Sleep time in seconds between each request.")
    args = parser.parse_args()
    config_file = str(args.config_file)
    model_name = str(args.model_name)
    query_method = str(args.query_method)
    temperature = float(args.temperature)
    algorithm = str(args.algorithm)    
    n_samples = int(args.n_samples)
    verbose = (True if args.verbose.lower()=="true" else False)
    sleep_time = int(args.sleep_time)

    # "Global" vars TODO: add argparse support
    task = 'generic-problem'
    file_query = './prompts/generic-problem.txt'
    # Other vars
    logs_folder = f'./logs/{model_name}/'

    # Create inputs
    p, x, y = get_problems(algorithm, n_samples)

    # Logs prefix
    os.makedirs(logs_folder, exist_ok=True)
    with open(f'{logs_folder}{task}-{query_method}.txt', 'a+') as file_:
        file_.write('#'*30 + '\n')
        file_.write(str(datetime.datetime.now()) + '\n')
        file_.write(f"algorithm: {algorithm}, n_samples: {n_samples}\n")
        
    for i, technique in enumerate(['vanilla', 'variation']):

        with open(f'{logs_folder}{task}-{query_method}.txt', 'a+') as file_:
            file_.write(f"technique: {algorithm}-{technique}\n")
        
        query = read_prompts(file_query, query_method)  # Get the query
        correct, total = 0, 0
        for v,gt in zip(x, y[i]):
            prompt = cp.deepcopy(query)
            prompt = prompt.replace('@input@', str(v))
            prompt = prompt.replace('@code@', str(p[i]))

            # Collect responses
            response = queryLLM(prompt, config_file, model_name, temperature)
            
            # Evaluate the response
            try:
                r = re.findall(r"<result>(.*)</result>", response)
                y_hat = r[-1]
            except:
                total += 1
                continue

            if algorithm == 'sort':
                y_hat = ast.literal_eval(y_hat)
            elif algorithm == 'prime':
                if '<result>true</result>' in response.lower():
                    y_hat = True
                elif '<result>false</result>' in response.lower():
                    y_hat = False
                else:
                    y_hat = -1
            else:
                try:
                    y_hat = int(y_hat)
                except:
                    continue

            if gt == y_hat:
                correct += 1

            total += 1
            sleep(sleep_time)  # sometimes the server crashes, so let's give it a break :) TODO: sync calls

            # Resuts
            if verbose:
                print(f"\n<prompt>\n{prompt}\n</prompt>\n")
                print(f"<response>\n{response}\n</response>\n")
                print(f"<ground-truth>{gt}</ground-truth>\n")

            # Logs
            with open(f'{logs_folder}{task}-{query_method}.txt', 'a+') as file_:
                file_.write(f"\n<prompt>\n{prompt}\n</prompt>\n")
                file_.write(f"<response>\n{response}\n</response>\n")
                file_.write(f"<ground-truth>{gt}</ground-truth>\n")

        with open(f'{logs_folder}{task}-{query_method}.txt', 'a+') as file_:
                file_.write(f"\n<accuracy>\n{correct/total}\n</accuracy>\n")


