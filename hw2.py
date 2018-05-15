# Starter code for CS 165B HW2
import numpy as np

def run_train_test(training_input, testing_input):
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition. 
    You are you are permitted to use the numpy library but you must write 
    your own code for the linear classifier. 

    Inputs:
        training_input: list form of the training file
            e.g. [[3, 5, 5, 5],[.3, .1, .4],[.3, .2, .1]...]
        testing_input: list form of the testing file

    Output:
        Dictionary of result values 

        IMPORTANT: YOU MUST USE THE SAME DICTIONARY KEYS SPECIFIED
        
        Example:
            return {
                "tpr": true_positive_rate,
                "fpr": false_positive_rate,
                "error_rate": error_rate,
                "accuracy": accuracy,
                "precision": precision
            }
    """

    # TODO: IMPLEMENT
    trn_info = training_input[0]
    tst_info = testing_input[0]

    trn = np.array(training_input[1:])
    tst = np.array(testing_input[1:])

    A_trn = trn[:trn_info[1]]
    B_trn = trn[trn_info[1]:trn_info[1]+trn_info[2]]
    C_trn = trn[trn_info[1]+trn_info[2]:]

    A_centroid = np.mean(A_trn, axis=0)
    B_centroid = np.mean(B_trn, axis=0)
    C_centroid = np.mean(C_trn, axis=0)

    AB_midpoint = np.mean(np.array([A_centroid, B_centroid]), axis = 0)
    BC_midpoint = np.mean(np.array([B_centroid, C_centroid]), axis = 0)
    AC_midpoint = np.mean(np.array([A_centroid, C_centroid]), axis = 0)

    A_tst = tst[:tst_info[1]]
    B_tst = tst[tst_info[1]:tst_info[1]+tst_info[2]]
    C_tst = tst[tst_info[1]+tst_info[2]:]

    # print trn.shape, tst.shape
    # print A_trn.shape, B_trn.shape, C_trn.shape
    # print A_tst.shape, B_tst.shape, C_tst.shape
    print A_centroid
    print B_centroid
    print C_centroid

    print AB_midpoint
    print BC_midpoint
    print AC_midpoint

    # return {
    #     "tpr": true_positive_rate,
    #     "fpr": false_positive_rate,
    #     "error_rate": error_rate,
    #     "accuracy": accuracy,
    #     "precision": precision
    # }


#######
# The following functions are provided for you to test your classifier.
######
def parse_file(filename):
    """
    This function is provided to you as an example of the preprocessing we do
    prior to calling run_train_test
    """
    with open(filename, "r") as f:
        data = [[float(y) for y in x.strip().split(" ")] for x in f]
        data[0] = [int(x) for x in data[0]]

        return data

if __name__ == "__main__":
    """
    You can use this to test your code.
    python hw2.py [training file path] [testing file path]
    """
    import sys

    training_input = parse_file(sys.argv[1])
    testing_input = parse_file(sys.argv[2])

    print run_train_test(training_input, testing_input)

