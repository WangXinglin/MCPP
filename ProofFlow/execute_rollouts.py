import json
import os
import re
import traceback

def clean_code(code):
    if code is None:
        return ""
    if isinstance(code, list):
        code = "\n".join(str(x) for x in code if x is not None)
    else:
        code = str(code)
    code = re.sub(r'```python\n|```\n|```', '', code)
    return code

def execute_rollouts(file_paths):
    print(f"| {'task':<15} | {'subtask':<15} | {'total':<5} | {'pass':<5} | {'rate':<7} | {'errors':<40} |")
    print(f"|{'-'*17}|{'-'*17}|{'-'*7}|{'-'*7}|{'-'*9}|{'-'*42}|")

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        
        task_name = os.path.basename(file_path)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

        subproblems = []
        if isinstance(data, list):
            subproblems = data
        elif isinstance(data, dict):
            subproblems = data.get('subproblems', data.get('problems', [data]))

        # Mapping for dependencies
        subproblems_map = {sp.get('name', sp.get('id')): sp for sp in subproblems}
        
        for sp in subproblems:
            subtask_name = sp.get('name', sp.get('id', 'N/A'))
            test_codes = sp.get('test_code', [])
            if not isinstance(test_codes, list):
                test_codes = [test_codes] if test_codes else []
            
            # Clean tests
            test_codes = [clean_code(t) for t in test_codes if clean_code(t).strip()]
            
            rollouts = sp.get('rollout_candidates', [])
            if not rollouts:
                rollouts = [sp.get('generated', '')]
            else:
                rollouts = [r.get('generated', '') if isinstance(r, dict) else r for r in rollouts]

            total_rollouts = 0
            pass_all_count = 0
            error_counts = {"no_tests": 0, "compile_error": 0, "dep_error": 0, "runtime_error": 0}

            if not test_codes:
                error_counts["no_tests"] = len(rollouts)
                # Still output a row even if no tests
                print(f"| {task_name[:15]:<15} | {str(subtask_name)[:15]:<15} | {0:<5} | {0:<5} | {'0.0%':<7} | no_tests: {len(rollouts)} |")
                continue

            for gen_raw in rollouts:
                gen_code = clean_code(gen_raw)
                if not gen_code.strip():
                    continue
                
                total_rollouts += 1
                namespace = {}
                failed = False

                # 3) Handle dependencies
                deps = sp.get('dependencies', [])
                for dep_name in deps:
                    dep_sp = subproblems_map.get(dep_name)
                    if dep_sp:
                        dep_gen = clean_code(dep_sp.get('generated', ''))
                        try:
                            exec(dep_gen, namespace)
                        except Exception:
                            error_counts["dep_error"] += 1
                            failed = True
                            break
                    else:
                        # Dependency not found - treat as error
                        error_counts["dep_error"] += 1
                        failed = True
                        break
                
                if failed:
                    continue

                # Execute generated code
                try:
                    exec(gen_code, namespace)
                except Exception:
                    error_counts["compile_error"] += 1
                    continue

                # 4) Run tests
                passed_all_tests = True
                for test_code in test_codes:
                    try:
                        exec(test_code, namespace)
                    except Exception:
                        passed_all_tests = False
                        error_counts["runtime_error"] += 1
                        break
                
                if passed_all_tests:
                    pass_all_count += 1

            rate = f"{(pass_all_count/total_rollouts)*100:.1f}%" if total_rollouts > 0 else "0.0%"
            err_str = ", ".join([f"{k}: {v}" for k, v in error_counts.items() if v > 0])
            print(f"| {task_name[:15]:<15} | {str(subtask_name)[:15]:<15} | {total_rollouts:<5} | {pass_all_count:<5} | {rate:<7} | {err_str[:40]:<40} |")

files = [
    "<PROJECT_ROOT>/codeflow/output/1657A (1).json",
    "<PROJECT_ROOT>/codeflow/output/1812I.json",
    "<PROJECT_ROOT>/codeflow/output/1866M.json"
]

execute_rollouts(files)
