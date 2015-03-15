import subprocess

print("{:.2f}% done".format(int(subprocess.check_output(['wc', '-l', 'test1.csv']).decode("utf-8").split(" ")[0]) / (200 * 2738) * 100))