import os
import json
import subprocess

prog_path = '/home/cctu/3dtranspose/test_inplace'
name_prefix = 'res_case'
input_file_name = 'fix_d2.json'

type_size = '4'
#testcase_path = '/home/cctu/testcase/peterwu/231case/fix_d3/'
#input_file_name = '231case_fix_d3.json'
#permutation = ['1', '2', '0'] # remember to -1 forall

def gen(case_list, testcase_path, permutation):
	print('\nVerifying testcases of ' + testcase_path)
	i = 1
	for c in case_list:
		#if i < 21:
		#	i += 1
		#	continue
		print('\n============ Case ' + str(i) + ' ============')
		prog = [prog_path]
		prog.extend(c)
		prog.extend(permutation)
		prog.append(type_size)
		output_file_name = testcase_path + name_prefix + str(i) + '.out'
		#prog.append(output_file_name)
		
		ans_file_name = testcase_path + 'ans_case' + str(i) + '.out'
		diff = ['diff', ans_file_name, output_file_name]
		rm = ['rm', '-f', output_file_name]
		#newline = ['echo', '\"\\n\"', '>>', 'inplace_bench.txt']
		
		#subprocess.run(newline)
		subprocess.run(prog)
		#output = subprocess.run(diff)
		#subprocess.run(rm)
		#if output.returncode > 0:
		#	return
		i += 1
		
def main():
	case_list = json.load(open(input_file_name))
	
	testcase_path = '/home/cctu/testcase/321case/fix_d2/'
	permutation = ['2', '1', '0']
	gen(case_list, testcase_path, permutation)

	#testcase_path = '/home/cctu/testcase/312case/fix_d2/'
	#permutation = ['2', '0', '1']
	#gen(case_list, testcase_path, permutation)

if __name__ == '__main__':
    main()
