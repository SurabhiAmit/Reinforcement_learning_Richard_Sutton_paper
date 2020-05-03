Implementation of the experiments in Richard Sutton(father of reinforcement learning)'s famous paper in which he introduces the concept of reinforcement learning to understand the nuances of this machine learning technique.

How to run the source code?

1. Open project1.py in IDE like PyCharm and click 'Run'.
2. Or, on command prompt, go to the location where project1.py is stored and type "project1.py" and hit Enter key.
2. The experiment 1 graph (Figure 3 in Sutton paper) gets generated.
3. On closing the graph, experiment 2 graph (Figure 4 in Sutton paper) gets generated.
4. On closing this graph window, the graph corresponding to Figure 5 in Sutton paper gets generated.

Note:

Imported libraries:

numpy, random, collections, matplotlib.

Code description:

The function "random_walk(lambda1, data)" performs the experiment 1 pertaining to Figure 3 in Sutton's paper. It calls the functions "update_from_set(set, w, lambda1)" to update the weight vector after a training set and "get_w_from_seq(seq, w, lambda1)" to obtain the change in weight vector due to each observation in a sequence. 
Similarly, the function "experiment2(lambda2, data2, alpha)" performs the experiment 2 pertaining to Figure 4 and Figure 5 in Sutton's paper.
The function "training_sets()" builds the required 100 training sets, each with 10 random walk sequences.

