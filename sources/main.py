import data
import tenfoldSVM
import bootstrappingSVM
import tenfoldKNN
import bootstrappingKNN

import sys


algorithms = sys.argv

data_info = False

if "datainfo" in algorithms:
    algorithms.remove("datainfo")
    data_info = True

X, y = data.extract(print_info=data_info)

for alg in algorithms:
    if alg == "bsknn":
        print("This should take under a minute - strapping neighbors together as we talk, sir/ma'am!\n")
        bootstrappingKNN.run(X, y, data_info)
    elif alg == "bssvm":
        print("This should take a minute or two - strapping in those support vectors as we talk, chief!\n")
        bootstrappingSVM.run(X, y, data_info)
    elif alg == "tfknn":
        print("This should take under a minute - folding neighbors together as we talk, el jefe!\n")
        tenfoldKNN.run(X, y, data_info)
    elif alg == "tfsvm":
        print("This should take a minute or two - folding in those support vectors as we talk, capt'n!\n")
        tenfoldSVM.run(X, y, data_info)
    elif alg == "svm":
        print("This can take up to 5 minutes - I'd recommend a nap to pass the time.\n")
        tenfoldSVM.run(X, y, data_info)
        bootstrappingSVM.run(X, y, data_info)
    elif alg == "knn":
        print("This should take 1-2 minutes - hold tight as we knock on some neighbors' doors for ya!\n")
        tenfoldKNN.run(X, y, data_info)
        bootstrappingKNN.run(X, y, data_info)
    elif alg == "all":
        print("This can take up 10 minutes - ambitious of you to commit to this.\n")
        bootstrappingKNN.run(X, y, data_info)
        bootstrappingSVM.run(X, y, data_info)
        tenfoldSVM.run(X, y, data_info)
        tenfoldKNN.run(X, y, data_info)
    elif alg == "tf":
        print("This should take 3-5 minutes - folding in process!\n")
        tenfoldSVM.run(X, y, data_info)
        tenfoldKNN.run(X, y, data_info)
    elif alg == "bs":
        print("This should take 3-5 minutes - strapping in process!\n")
        bootstrappingSVM.run(X, y, data_info)
        bootstrappingKNN.run(X, y, data_info)
