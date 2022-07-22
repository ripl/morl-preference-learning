# FOR TESTING PURPOSES ONLY
import os

if __name__=="__main__":
    print("before file")
    with open('/code/src/morl-preference-learning/packages/PbMORL/models/test.txt', 'w+') as f:
        f.write('testing...')
    print("after file")
    with open('./packages/PbMORL/models/test.txt', 'r') as f:
        print(f.read())