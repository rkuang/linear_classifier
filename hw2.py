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

    # Get number of examples per class
    trn_info = training_input[0]
    tst_info = testing_input[0]

    # discount the metadata
    trn = np.array(training_input[1:])
    tst = np.array(testing_input[1:])

    # separate the training data
    A_trn = trn[:trn_info[1]]
    B_trn = trn[trn_info[1]:trn_info[1]+trn_info[2]]
    C_trn = trn[trn_info[1]+trn_info[2]:]

    # calculate the centroids
    A_centroid = np.mean(A_trn, axis=0)
    B_centroid = np.mean(B_trn, axis=0)
    C_centroid = np.mean(C_trn, axis=0)

    # calculate the coefficients of the functions
    AB_coef = A_centroid - B_centroid
    BC_coef = B_centroid - C_centroid
    AC_coef = A_centroid - C_centroid

    # calculate the midpoints of the centroids
    AB_midpoint = np.mean(np.array([A_centroid, B_centroid]), axis = 0)
    BC_midpoint = np.mean(np.array([B_centroid, C_centroid]), axis = 0)
    AC_midpoint = np.mean(np.array([A_centroid, C_centroid]), axis = 0)

    # store discriminant functions as arrays
    AB_classifier = np.append(AB_coef, -1*np.dot(AB_coef, AB_midpoint))
    BC_classifier = np.append(BC_coef, -1*np.dot(BC_coef, BC_midpoint))
    AC_classifier = np.append(AC_coef, -1*np.dot(AC_coef, AC_midpoint))

    confusion = np.zeros(16).reshape(4,4)
    for i in range(3):
        confusion[3,i] = tst_info[i+1]
    confusion[3,3] = np.sum(confusion[3,:])

    A_tst = tst[:tst_info[1]]
    B_tst = tst[tst_info[1]:tst_info[1]+tst_info[2]]
    C_tst = tst[tst_info[1]+tst_info[2]:]

    for example in tst:
        AB_score = np.dot(AB_classifier[:tst_info[0]], example) + AB_classifier[tst_info[0]]
        if AB_score >= 0:
            AC_score = np.dot(AC_classifier[:tst_info[0]], example) + AC_classifier[tst_info[0]]
            if AC_score >= 0:
                # predict A
                confusion[0,3] += 1
                if example in A_tst:
                    confusion[0,0] += 1
                elif example in B_tst:
                    confusion[0,1] += 1
                else:
                    confusion[0,2] += 1
            else:
                # predict C
                confusion[2,3] += 1
                if example in C_tst:
                    confusion[2,2] += 1
                elif example in A_tst:
                    confusion[2,0] += 1
                else:
                    confusion[2,1] += 1
        else:
            BC_score = np.dot(BC_classifier[:tst_info[0]], example) + BC_classifier[tst_info[0]]
            if BC_score >= 0:
                # predict B
                confusion[1,3] += 1
                if example in B_tst:
                    confusion[1,1] += 1
                elif example in A_tst:
                    confusion[1,0] += 1
                else:
                    confusion[1,2] += 1
            else:
                # predict C
                confusion[2,3] += 1
                if example in C_tst:
                    confusion[2,2] += 1
                elif example in A_tst:
                    confusion[2,0] += 1
                else:
                    confusion[2,1] += 1
                
        i += 1
    print confusion

    tpr_A = confusion[0,0]/confusion[3,0]
    tpr_B = confusion[1,1]/confusion[3,1]
    tpr_C = confusion[2,2]/confusion[3,2]
    true_positive_rate = (tpr_A+tpr_B+tpr_C)/3

    fpr_A = (confusion[0,3]-confusion[0,0])/(confusion[3,3]-confusion[3,0])
    fpr_B = (confusion[1,3]-confusion[1,1])/(confusion[3,3]-confusion[3,1])
    fpr_C = (confusion[2,3]-confusion[2,2])/(confusion[3,3]-confusion[3,2])
    false_positive_rate = (fpr_A+fpr_B+fpr_C)/3

    error_A = (fpr_A*(confusion[3,3]-confusion[3,0])+(1-tpr_A)*confusion[3,0])/confusion[3,3]
    error_B = (fpr_B*(confusion[3,3]-confusion[3,1])+(1-tpr_B)*confusion[3,1])/confusion[3,3]
    error_C = (fpr_C*(confusion[3,3]-confusion[3,2])+(1-tpr_C)*confusion[3,2])/confusion[3,3]
    error_rate = (error_A+error_B+error_C)/3

    a_A = (confusion[3,0]*tpr_A+(confusion[3,1]+confusion[3,2])*(1-fpr_A))/confusion[3,3]
    a_B = (confusion[3,1]*tpr_B+(confusion[3,0]+confusion[3,2])*(1-fpr_B))/confusion[3,3]
    a_C = (confusion[3,2]*tpr_C+(confusion[3,0]+confusion[3,1])*(1-fpr_C))/confusion[3,3]
    accuracy = (a_A+a_B+a_C)/3

    p_A = confusion[0,0]/confusion[0,3]
    p_B = confusion[1,1]/confusion[1,3]
    p_C = confusion[2,2]/confusion[2,3]
    precision = (p_A+p_B+p_C)/3

    return {
        "tpr": true_positive_rate,
        "fpr": false_positive_rate,
        "error_rate": error_rate,
        "accuracy": accuracy,
        "precision": precision
    }

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

