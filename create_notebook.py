import IPython.nbformat.current as nbf

nb = nbf.read(open('Code/Read_Data_Pandas.py', 'r'), 'py')
nbf.write(nb, open('Code/Read_Data_Pandas.ipynb', 'w'), 'ipynb')
