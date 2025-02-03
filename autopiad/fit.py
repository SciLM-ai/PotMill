import numpy as np
import pandas as pd
import random
from autopiad.tools import rcuts_to_string, twojmaxes_to_string


def fit(features_directory, hyperparameters, feature_names, train_fraction = 0.7, n_fold = 3, rcond = 1e-10):

    # print("CPU count is",multiprocessing.cpu_count(),os.cpu_count())
    rcuts, twojmaxes, eweight = hyperparameters
    feature_indices = [i for i, lst in enumerate(feature_names) if len(lst)==1 or all(value <= twojmaxes[0] for value in lst[1:])]
    b_vect = pd.read_csv(features_directory + "b.csv", index_col=0, header=None).sort_index()
    b_vect_index = b_vect.index.to_numpy()
    a_matr_map = np.load(features_directory + rcuts_to_string(rcuts,delimiter="_") + "/a.npy", mmap_mode='r')
    a_matr = a_matr_map[b_vect_index[:, None],feature_indices]

    b_vect.reset_index(inplace=True)
    b_vect_no_dupl = b_vect.drop_duplicates(subset=1)
    job_ids = b_vect_no_dupl[1].to_list()
    energy_selector = b_vect_no_dupl.index.to_list()
    force_selector = b_vect.drop(index=energy_selector).index.to_list()
    assert len(force_selector)+len(energy_selector) == b_vect.shape[0]

    # log_file = fit_directory + "fit_report_rcut_%.3f_2jmax_%i.txt" % hyperparameters
    # with open(log_file, 'w') as sys.stdout:

    for fold in range(n_fold):

        print("===================== FOLD ",fold," OF ",n_fold,"=====================")
        print("Hyperparameters rcut, 2Jmax and eweight are " + rcuts_to_string(rcuts) + ", " + 
              twojmaxes_to_string(twojmaxes) + " and %.3f"%eweight)
        
        random.seed(fold)
        random.shuffle(job_ids)
        train_ids = job_ids[:int(train_fraction*len(job_ids))]
        test_ids = job_ids[int(train_fraction*len(job_ids)):]
        train_index = b_vect[b_vect[1].isin(train_ids)].index.to_list()
        test_index = b_vect[b_vect[1].isin(test_ids)].index.to_list()

        a_train = a_matr[train_index]
        a_test = a_matr[test_index]
        b_train = b_vect[2].values[train_index]
        b_test = b_vect[2].values[test_index]
        
        energy_selector_train = np.where(np.in1d(train_index, energy_selector))[0]
        energy_selector_test = np.where(np.in1d(test_index, energy_selector))[0]
        force_selector_train = np.where(np.in1d(train_index, force_selector))[0]
        force_selector_test = np.where(np.in1d(test_index, force_selector))[0]
        # energy_selector_train = [i for i in range(len(train_index)) if train_index[i] in energy_selector]
        # energy_selector_test = [i for i in range(len(test_index)) if test_index[i] in energy_selector]
        # force_selector_train = [i for i in range(len(train_index)) if train_index[i] in force_selector]
        # force_selector_test = [i for i in range(len(test_index)) if test_index[i] in force_selector]

        eweights_train = np.exp(-b_train[energy_selector_train]/5)
        eweights_train /= np.sum(eweights_train)
        eweights_train *= eweight

        fweights_train = 1./np.maximum(3.,np.fabs(b_train[force_selector_train]))
        fweights_train /= np.sum(fweights_train)
        fweights_train *= 1.

        eweights_test = np.exp(-b_test[energy_selector_test]/5)
        eweights_test /= np.sum(eweights_test)
        eweights_test *= 1.

        fweights_test = 1./np.maximum(3.,np.fabs(b_test[force_selector_test]))
        fweights_test /= np.sum(fweights_test)
        fweights_test *= 1.

        a_e_train_w = np.multiply(eweights_train[:,None],a_train[energy_selector_train])
        a_f_train_w = np.multiply(fweights_train[:,None],a_train[force_selector_train])
        b_e_train_w = np.multiply(eweights_train,b_train[energy_selector_train])
        b_f_train_w = np.multiply(fweights_train,b_train[force_selector_train])
        print(a_e_train_w.shape, a_f_train_w.shape)

        a_stack = np.concatenate([a_e_train_w,a_f_train_w])
        b_stack = np.concatenate([b_e_train_w,b_f_train_w])
        print(a_stack.shape, b_stack.shape)
        
        beta, *_ = np.linalg.lstsq(a_stack, b_stack, rcond)

        train_residual = np.square(np.dot(a_train,beta) - b_train)
        train_e_rmse = np.sqrt(np.mean(train_residual[energy_selector_train]))
        train_f_rmse = np.sqrt(np.mean(train_residual[force_selector_train]))
        train_e_rmse_weighted = np.sqrt(np.sum(np.multiply(eweights_train,train_residual[energy_selector_train])))
        train_f_rmse_weighted = np.sqrt(np.sum(np.multiply(fweights_train,train_residual[force_selector_train])))
        print("Energy training RMSE is", np.sqrt(np.mean(train_residual[energy_selector_train])))
        print("Force training RMSE is", np.sqrt(np.mean(train_residual[force_selector_train])))
        test_residual = np.square(np.dot(a_test,beta) - b_test)
        test_e_rmse = np.sqrt(np.mean(test_residual[energy_selector_test]))
        test_f_rmse = np.sqrt(np.mean(test_residual[force_selector_test]))
        test_e_rmse_weighted = np.sqrt(np.sum(np.multiply(eweights_test,test_residual[energy_selector_test])))
        test_f_rmse_weighted = np.sqrt(np.sum(np.multiply(fweights_test,test_residual[force_selector_test])))
        print("Energy testing RMSE is", np.sqrt(np.mean(test_residual[energy_selector_test])))
        print("Force testing RMSE is", np.sqrt(np.mean(test_residual[force_selector_test])))
        

        with open("results.csv","a") as file:
            results_line = "%i,"%fold + rcuts_to_string(rcuts,delimiter=",") + "," + twojmaxes_to_string(twojmaxes,delimiter=",")
            results_line += ",%.3f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f\n" % \
                    (eweight,train_e_rmse,train_f_rmse,test_e_rmse,test_f_rmse,train_e_rmse_weighted,
                     train_f_rmse_weighted,test_e_rmse_weighted,test_f_rmse_weighted)
            file.write(results_line)
        
        beta_filename = "pot__rcut_" + rcuts_to_string(rcuts,delimiter="_") + "__2jmax_" + \
            twojmaxes_to_string(twojmaxes,delimiter="_") + "__eweight_%.3f__fold_%i.csv" % (eweight,fold) 
        np.savetxt(beta_filename, beta)

        # bins={}
        # bins['e']=0.5
        # bins['f']=2
        # binned_errors=compute_binned_errors(beta,train_a,train_b,test_a,test_b,bins)

        # print("Hyperparameters rcut, 2Jmax and eweight are", rcut, twojmax, eweight)
        # print("Errors", errors)
        # print("Binned errors", binned_errors)

    # print("Time elapsed: %.5f seconds" % (time.time()-start_time))