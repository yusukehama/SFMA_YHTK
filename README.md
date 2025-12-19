# Subsampling Factorzation Machine Annealing (SFMA)

This repository contains a series of files for performing computations for the algorithms described in this paper. It is presented to demonstrate the reproducibility of our research and share our results and insights with the broader community. We hope that it is useful for related works and serves as a foundation for future extensions.   

---

## File Structure

SFMA_YHTK/
├── parameters_SFMA.py
├── functions_SFMA.py
├── check_SA.ipynb
├── check_QA.ipynb
├── initialdata_SA.ipynb
├── initialdata_QA.ipynb
├── input_SA_standard.ipynb
├── input_SA_nonstandard.ipynb
├── input_singleR_SA_standard.ipynb
├── input_extended_SA_standard.ipynb
├── input_extended_SA_nonstandard.ipynb
├── input_extended_singleR_SA_standard.ipynb
├── input_extended_multipleR_SA_standard.ipynb
├── input_RS.ipynb
├── input_extended_RS.ipynb
├── input_QA_standard.ipynb
├──output1_check_SA.ipynb
├──output2_check_SA.ipynb
├──output3_check_SA.ipynb
├──output1_check_QA.ipynb
├──output2_check_QA.ipynb
├──output3_check_QA.ipynb
├──data_plot_12_SA.ipynb
├──data_plot_16_SA.ipynb
├──data_plot_20_SA.ipynb
├──extended_data_plot_12_SA.ipynb
├──extended_data_plot_16_SA.ipynb
├──extended_data_plot_20_SA.ipynb
├──data_plot_12_SA_and_QA.ipynb
├──data_plot_16_SA_and_QA.ipynb
├──data_plot_20_SA_and_QA.ipynb
├──final_data_file_12/
├──final_data_file_16/
├──final_data_file_20/
├──final_SFMA_results_mean_12_SA/
├──final_SFMA_results_rate_12_SA/
├──final_SFMA_results_ext_mean_12_SA/
├──final_SFMA_results_ext_rate_12_SA/
├──final_SFMA_results_mean_12_SA_and_QA/
├──final_SFMA_results_rate_12_SA_and_QA/
├──final_SFMA_results_mean_16_SA/
├──final_SFMA_results_rate_16_SA/
├──final_SFMA_results_ext_mean_16_SA/
├──final_SFMA_results_ext_rate_16_SA/
├──final_SFMA_results_mean_16_SA_and_QA/
├──final_SFMA_results_rate_16_SA_and_QA/
├──final_SFMA_results_mean_20_SA/
├──final_SFMA_results_rate_20_SA/
├──final_SFMA_results_ext_mean_20_SA/
├──final_SFMA_results_ext_rate_20_SA/
├──final_SFMA_results_mean_20_SA_and_QA/
├──final_SFMA_results_rate_20_SA_and_QA/
├── README.md
└── LICENSE
└── requirements.txt



## Requirements
All required Python packages are listed in `requirements.txt`.
Install them via:

pip install -r requirements.txt  

## Components

### 1. File Listing the Parameters for Executing SFMA and FMA 
1. parameters_SFMA.py: - The notebook which lists all the parameters for performing computations.  


### 2. File Listing the Functions for Executing SFMA and FMA
1. functions_SFMA.py: - The notebook which lists all the functions for performing computations.
  

### 3. Files for Verifying the Validity of SA and QA 
1. check_SA.ipynb: - The notebook for verifying the validity of simulated annealing (SA) through FM models which are trained using the inital datasets D_0 under two conditions: with and without standardization. The value of n_bit can be controlled by tuning that of a parameter num_N through a relation n_bit = num_N*num_K, where num_K=2 (see "parameters_SFMA.py"). 
   
2. check_QA.ipynb: - The notebook for verifying the validity of quanutm annealing (QA) through FM models which are trained using the inital datasets D_0 with standardization. n_bit can be controlled by tuning num_N. The parameter num_W specifies the index of a target W matrix.

### D-Wave API Token Setup
To run this code with D-Wave's quantum computer, you need an API token.
1. Obtain your token from D-Wave Leap.
2. Set it as an environment variable (see functions_SFMA.py):
   ```bash
   export my_token = 'my_token'                 

### 4. Files for Generating Initial Datasets 
1. initaldata_SA.ipynb: - The notebook for generating the original initial datasets D_0 and the augumented inital datasets D_1 by SA. n_bit can be controlled by tuning num_N.  　　　
                               　　　　　　　　　　　　　　　 
2. initaldata_QA.ipynb: - The notebook for generating D_0 and D_1 by QA. n_bit can be controlled by tuning num_N. The parameter num_W is the same one used in "check_QA.ipynb".  


### 5. Files for Executing SFMA and FMA
1. input_SA_standard.ipynb: - The notebook for executing SFMA and FMA using SA together with standaridization. The parameter num_N are the same one used in "initaldata_SA.ipynb". The parameters denoted by nrep_it, n_samp, and "ratio", describes the number of iterations of black-box optimization, the number of samples of datasets, and the ratio characterizing the sizes of subdatasets, respectively. The paramters num_W_a and num_W_b run from 0 to 9 and num_W_a is less than num_W_b. By specifying the values of num_W_a and num_W_b, the file generates the results for the num_W_a-th to the num_W_b-th W matrices. When num_W_b = num_W_a + 1 = n + 1, the file generates the results for the n-th W matrix.                         
     
2. input_SA_nonstandard.ipynb: - The notebook for executing SFMA and FMA using SA without standaridization. The parameters num_N, nrep_it, n_samp, "ratio", num_W_a, and num_W_b have the same meanings with those used in "input_SA_standard.ipynb".   

3. input_QA_standard.ipynb: - The notebook for executing SFMA using QA together with standaridization. The parameters num_N, nrep_it, n_samp, and "ratio" have the same meanings with those used in "input_SA_standard.ipynb". Furthermore, the parameter num_W has the same meaning with that used in "check_QA.ipynb" or "initaldata_QA.ipynb".    
                                                             
4. input_singleR_SA_standard.ipynb: - The notebook for executing SFMA using SA together with standaridization. The parameters num_N, nrep_it, n_samp, and "ratio" have the same meanings with those used in "input_SA_standard.ipynb". Furthermore, the parameter num_W specifies the index of a target W matrix.        
                                      
5. input_extended_SA_standard.ipynb: - The notebook for executing SFMA and FMA using SA together with standaridization. The output datasets are generated by augumenting the datasets obtained by "input_SA_standard.ipynb" with those yielded by the additional black-box optimization loops given by the parameter nrep_it_ext and ratio=0.4. The parameters num_N, nrep_it, n_samp, and num_W have the same meanings with those used in "input_singleR_SA_standard.ipynb".  
                                                                                                                        
6. input_extended_SA_nonstandard.ipynb: - The notebook for executing SFMA and FMA using SA without standaridization. The output datasets are generated by augumenting the datasets obtained by "input_SA_nonstandard.ipynb" with those yielded by the additional black-box optimization loops given by the parameter nrep_it_ext. The parameters num_N, nrep_it, n_samp, "ratio", and num_W have the same meanings with those used in "input_extended_SA_standard.ipynb".  
                                          
7. input_extended_singleR_SA_standard.ipynb: - The notebook for executing SFMA using SA together with standaridization. The output datasets are generated by augumenting the datasets obtained by the file "input_singleR_SA_standard.ipynb" with those yielded by the additional black-box optimization loops given by the parameter nrep_it_ext. The parameters num_N, nrep_it, n_samp, "ratio", and num_W have the same meanings with those used in "input_singleR_SA_standard.ipynb". 
                                               
8. input_extended_multipleR_SA_standard.ipynb: - The notebook for executing SFMA using SA together with standaridization. The output datasets are generated by augumenting the datasets obtained by "input_singleR_SA_standard.ipynb" with those yielded by the additional black-box optimization loops given by the parameters nrep_it_ext and ratio_ext. The parameters num_N, nrep_it, n_samp, "ratio", and num_W have the same meanings with those used in "input_extended_singleR_SA_standard.ipynb". 

9. input_RS.ipynb: - The notebook for executing random search (RS). The output datasets are created by incorporating the initial datasets D_0 into the datasets generated by RS given by the parameter nrep_it. 
                        
10. input_extended_RS.ipynb: - The notebook for executing RS. The output datasets are created by incorporating the datasets obtained by "input_RS.ipynb" into those generated by RS given by the parameter nrep_it_ext.   
                                         

### 6. Directories Containing Save the Inital Datasets and the Results
1. final_data_file_12: - The directory containing the inital datasets and the results for n_bit = 12.

2. final_data_file_16: - The directory containing the inital datasets and the results for n_bit = 16.

3. final_data_file_20: - The directory containing the inital datasets and the results for n_bit = 20. 

4. final_SFMA_results_mean_12_SA: - The directory containing the files of the plots (png files) for the results of mean by SA for n_bit = 12 and n_tot = 2*n_bit^2 + 1.
 
5. final_SFMA_results_rate_12_SA: - The directory containing the files of the plots for the results of success rate by SA for n_bit = 12 and n_tot = 2*n_bit^2 + 1.

6. final_SFMA_results_ext_mean_12_SA: - The directory containing the files of the plots for the results of mean by SA for n_bit = 12 and n_tot = 4*n_bit^2 + 1.

7. final_SFMA_results_ext_rate_12_SA: - The directory containing the files of the plots for the results of success rate by SA for n_bit = 12 and n_tot = 4*n_bit^2 + 1.
 
8. final_SFMA_results_mean_12_SA_and_QA: - The directory containing the file of the plot for the results of mean by SA and QA for n_bit = 12 and n_tot = 2*n_bit^2 + 1.
 
9. final_SFMA_results_rate_12_SA_and_QA: - The directory containing the file of the plot for the results of success rate by SA and QA for n_bit = 12 and n_tot = 2*n_bit^2 + 1.
 
10. final_SFMA_results_mean_16_SA: - The directory containing the files of the plots (png files) for the results of mean by SA for n_bit = 16 and n_tot = 2*n_bit^2 + 1.
 
11. final_SFMA_results_rate_16_SA: - The directory containing the files of the plots for the results of success rate by SA for n_bit = 16 and n_tot = 2*n_bit^2 + 1.

12. final_SFMA_results_ext_mean_16_SA: - The directory containing the files of the plots for the results of mean by SA for n_bit = 16 and n_tot = 4*n_bit^2 + 1.

13. final_SFMA_results_ext_rate_16_SA: - The directory containing the files of the plots for the results of success rate by SA for n_bit = 16 and n_tot = 4*n_bit^2 + 1.
 
14. final_SFMA_results_mean_16_SA_and_QA: - The directory containing the file of the plot for the results of mean by SA and QA for n_bit = 16 and n_tot = 2*n_bit^2 + 1.
 
15. final_SFMA_results_rate_16_SA_and_QA: - The directory containing the file of the plot for the results of success rate by SA and QA for n_bit = 16 and n_tot = 2*n_bit^2 + 1.

16. final_SFMA_results_mean_20_SA: - The directory containing the files of the plots (png files) for the results of mean by SA for n_bit = 20 and n_tot = 2*n_bit^2 + 1.
 
17. final_SFMA_results_rate_20_SA: - The directory containing the files of the plots for the results of success rate by SA for n_bit = 20 and n_tot = 2*n_bit^2 + 1.

18. final_SFMA_results_ext_mean_20_SA: - The directory containing the files of the plots for the results of mean by SA for n_bit = 20 and n_tot = 6*n_bit^2 + 1.

19. final_SFMA_results_ext_rate_20_SA: - The directory containing the files of the plots for the results of success rate by SA for n_bit = 20 and n_tot = 6*n_bit^2 + 1.
 
20. final_SFMA_results_mean_20_SA_and_QA: - The directory containing the file of the plot for the results of mean by SA and QA for n_bit = 20 and n_tot = 2*n_bit^2 + 1.
 
21. final_SFMA_results_rate_20_SA_and_QA: - The directory containing the file of the plot for the results of success rate by SA and QA for n_bit = 20 and n_tot = 2*n_bit^2 + 1.


### 7. Files for Displaying the Results
1. output1_check_SA.ipynb: The notebook which display the results of the validity of SA for n_bit = 12 obtained by executing "check_SA.ipynb".

2. output2_check_SA.ipynb: The notebook which display the results of the validity of SA for n_bit = 16 obtained by executing "check_SA.ipynb".

3. output3_check_SA.ipynb: The notebook which display the results of the validity of SA for n_bit = 20 obtained by executing "check_SA.ipynb".

4. output1_check_QA.ipynb: The notebook which display the results of the validity of QA for n_bit = 12 obtained by executing "check_QA.ipynb".

5. output2_check_QA.ipynb: The notebook which display the results of the validity of QA for n_bit = 16 obtained by executing "check_QA.ipynb".

6. output3_check_QA.ipynb: The notebook which display the results of the validity of QA for n_bit = 20 obtained by executing "check_QA.ipynb".

7. data_plot_12_SA.ipynb: - The notebook for plotting the results for n_bit = 12 and n_tot = 2*n_bit^2 + 1 obtained by "input_SA_standard.ipynb", "input_SA_nonstandard.ipynb", and "input_RS.ipynb".
 
8. data_plot_16_SA.ipynb: - The notebook for plotting the results for n_bit = 16 and n_tot = 2*n_bit^2 + 1 obtained by "input_SA_standard.ipynb", "input_SA_nonstandard.ipynb", and "input_RS.ipynb". 
 
9. data_plot_20_SA.ipynb: - The notebook for plotting the results for n_bit = 20 and n_tot = 2*n_bit^2 + 1 obtained by "input_SA_standard.ipynb", "input_SA_nonstandard.ipynb", and "input_RS.ipynb". 

10. extended_data_plot_12_SA.ipynb: - The notebook for plotting the results for n_bit = 12 and n_tot = 4*n_bit^2 + 1 obtained by "input_extended_SA_standard", "input_extended_SA_nonstandard", "input_extended_singleR_SA_standard.ipynb", and  "input_extended_RS.ipynb". 
                                     
11. extended_data_plot_16_SA.ipynb: - The notebook for plotting the results for n_bit = 16 and n_tot = 4*n_bit^2 + 1 obtained by "input_extended_SA_standard", "input_extended_SA_nonstandard", "input_extended_singleR_SA_standard.ipynb", and  "input_extended_RS.ipynb". 

12. extended_data_plot_20_SA.ipynb: - The notebook for plotting the results for n_bit = 20 and n_tot = 6*n_bit^2 + 1 obtained by "input_extended_SA_standard", "input_extended_SA_nonstandard", "input_extended_singleR_SA_standard.ipynb", "input_extended_multipleR_SA_standard.ipynb", and "input_extended_RS.ipynb".  

13. data_plot_12_SA_and_QA.ipynb: - The notebook for plotting the results for n_bit = 12 and n_tot = 2*n_bit^2 + 1 obtained by "input_SA_standard.ipynb" and "input_QA_standard". 

14. data_plot_16_SA_and_QA.ipynb: - The notebook for plotting the results for n_bit = 16 and n_tot = 2*n_bit^2 + 1 obtained by "input_SA_standard.ipynb" and "input_QA_standard". 
                                  
15. data_plot_20_SA_and_QA.ipynb: - The notebook for plotting the results for n_bit = 20 and n_tot = 2*n_bit^2 + 1 obtained by "input_SA_standard.ipynb" and "input_QA_standard". 

The data files of the W matrices used in this research (“C6_6x2.pickle.bz2”, “C6_8x2.pickle.bz2”, and “C6_10x2.pickle.bz2”: see, for instance, "initialdata_SA.ipynb") can be generated using files "z000.ipynb" and "mid.py", which can be obtained from Supplementary Information 2 and Supplementary Information 3, respectively, in Ref. [86] (DOI: https://doi.org/10.1038/s41598-022-19763-8). According to the restrictions of the applied licenses, these files are not redistributed in this repository. The data files are created by changing the set of parameters (N, D, K) in "z000.ipynb" from (8, 100, 3) to (6, 50, 2), (8, 50, 2), and (10, 50, 2) for n_bit = 12,16, and 20, respectively. In addition, the original data of the W matrices, the model parameters of the final fully connected layer of VGG16 convolutional neural network (Ref. [109]), is necessary. The data file of the model parameters used in this research was obtained externally and is not redistributed in this repository. Please download them using standard libraries such as torchvision.  

## How to Reproduce the Results
To reproduce the results, follow the steps below in order:
1. The valiation of SA – Execute "check_SA.ipynb" by setting num_N = 6,8, and 10 for n_bit = 12,16, and 20, respectively.
   
2. The valiation of QA – Execute "check_QA.ipynb" by setting (num_N, num_W) = (6,3),(8,4), and (10,2) for n_bit = 12,16, and 20, respectively.
   
3. Preparation of inital datasets by SA – Execute "initaldata_SA.ipynb" and generate D_0 and D_1 by setting num_N = 6,8, and 10 for n_bit = 12,16, and 20, respectively.
   
4. Preparation of inital datasets by QA – Execute "initaldata_QA.ipynb" and generate D_0 and D_1 by setting (num_N, num_W) = (6,3),(8,4), and (10,2) for n_bit = 12,16, and 20, respectively.
   
5. Execution of the standardized SMFA and FMA by SA with "ratio" = 0.4 – Execute "input_SA_standard.ipynb" by setting the parameters nrep_it, n_samp, "ratio", num_W_a, and num_W_b to 2, 30, 0.4, 0, and 9, respectively. 
   
6. Execution of the non-standardized SMFA and FMA by SA with "ratio" = 0.4 – Execute "input_SA_nonstandard.ipynb". The parameters num_N, nrep_it, n_samp, "ratio", num_W_a, and num_W_b should be set to the same values used in "input_SA_standard.ipynb".
   
7. Execution of the standardized SMFA by SA with "ratio" = 0.1 – Execute "input_singleR_SA_standard.ipynb" by setting n_samp and "ratio" to 30, and 0.1, respectively. The parameter num_W should be set to 2,5, and 7 for num_N = 6 and 8 with nrep_it = 4. For num_N = 10, nrep_it should be set to 6 for num_W = 3,5, and 7 while nrep_it = 5, 4, and 1 for num_W = 1,4, and 9, respectively.   
    
8. Execution of the standardized SMFA by QA with "ratio" = 0.4 – Execute "input_QA_standard.ipynb" by setting nrep_it, n_samp, and "ratio" to 2, 30, and 0.4, respectively. Furthermore, the set of parameters (num_N, num_W) should be set to the values used in "initaldata_QA.ipynb".
    
9. Execution of the extended standardized SMFA and FMA with "ratio" = 0.4 – Execute "input_extended_SA_standard.ipynb" by setting nrep_it, n_samp, and "ratio" to 2, 30, and 0.4, respectively. The parameter num_W should be set to the same values used in "input_singleR_SA_standard.ipynb". Furthermore, the set of parameters (num_N, nrep_it_ext) should be set to (6,2), (8,2), and (10,4) for n_bit = 12, 16, and 20, resepctively. 
    
10. Execution of the extended non-standardized SMFA and FMA with "ratio" = 0.4 – Execute "input_extended_SA_nonstandard.ipynb". The parameters num_N, nrep_it, nrep_it_ext, n_samp, "ratio", and num_W should be set to the same values used in "input_extended_SA_standard.ipynb".
    
11. Execution of the extended standardized SMFA with "ratio" = 0.1 – Execute "input_extended_singleR_SA_standard.ipynb" with setting num_N, n_samp, and "ratio" to = 10, 30, and 0.1, respectively. The set of parameters (nrep_it, nrep_it_ext) should be set to (5,1), (4,2), and (1,5) for num_W = 1,4, and 9, respectively. 
    
12. Execution of the extended standardized SMFA with mutiple ratios – Execute "input_extended_multipleR_SA_standard.ipynb" with setting num_N = 10 and ratio_ext = 0.01. The parameters nrep_it, nrep_it_ext, n_samp, "ratio", and num_W should be set to the same values used in "input_extended_singleR_SA_standard.ipynb". 
    
13. Execution of RS – Execute "input_extended_RS.ipynb". The parameters num_N, nrep_it, n_samp, num_W_a, and num_W_b should be set to the same values used in "input_SA_standard.ipynb".
    
14. Execution of the extended RS – Execute "input_extended_RS.ipynb". The parameters num_N, nrep_it, nrep_it_ext, n_samp, and num_W should be set to the same values used in "input_extended_SA_standard.ipynb".

## LICENSE
This project is licensed under the Apache License 2.0.  
See the LICENSE file for details.

## Citation
This code accompanies the paper: 

Paper Title: Subsampling Factorization Machine Annealing  
Author(s): Yusuke Hama and Tadashi Kadowaki 
Journal Name: Physical Review Research  
Year: (to be added after publication)   
DOI: (to be added after publication)

Once the paper is published, the DOI and citation information will be updated here.
