import os

import sys


# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_target_problems import get_full_CoT_token_ranges

if __name__ == "__main__":
    problem_num = 468201
    problem_dir = os.path.join("target_problems", "temperature_0.6_top_p_0.95")
    sentence2ranges, problem = get_full_CoT_token_ranges(
        problem_num, problem_dir, only_pre_convergence="semi", verbose=True
    )
    # print(problem["base_solution"]["full_cot"])
