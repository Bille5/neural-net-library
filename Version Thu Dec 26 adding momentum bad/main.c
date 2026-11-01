
#include <stdio.h>
#include "px5_neural_network.h"


int  xor_layer_specification[] = { 2,3,1,0};

float  test_case_1[] = { 0.0, 0.0 };
float  test_case_2[] = { 0.0, 1.0 };
float  test_case_3[] = { 1.0, 0.0 };
float  test_case_4[] = { 1.0, 1.0 };

float  test_case_1_expected[] = { 0.0 };
float  test_case_2_expected[] = { 1.0 };
float  test_case_3_expected[] = { 1.0 };
float  test_case_4_expected[] = { 0.0 };

float  results[1];

neural_network_handle    xor_network;


int main()
{

	float test_error;
	float test_pass_error;
	int   total_passes = 0;
	int   still_learning;
	int x;
	float percent_converge;
	int average_runs;
	int complete_passes;

	printf("Neural Network XOR test example!\n");
	px5_neural_network_create(&xor_network, xor_layer_specification);
	px5_neural_network_print(xor_network);
	complete_passes = 0;
	percent_converge = 0;
	average_runs = 0;

	for (x = 0; x < 1; x++)
	{
		/*if (x % 10 == 0)
		{
			printf("%.1f%% done\n", ((((float)x) / ((float)10000)) * 100));
		}
		total_passes = 0;*/
		/* Loop to train the network  */
		do
		{
			/*if (total_passes > 2000000)
			{
				percent_converge = percent_converge - 1;
				break;
			}*/
			total_passes++;
			test_pass_error = 0;
			still_learning = 0;

			/* Compute the first test case.  */
			px5_neural_network_compute(xor_network, test_case_1, 2, results, 1);

			/* Calculate the error.  */
			if (results[0] > test_case_1_expected[0])
				test_error = results[0] - test_case_1_expected[0];
			else
				test_error = test_case_1_expected[0] - results[0];

			/* Keep track of the error for each pass.  */
			test_pass_error = test_pass_error + test_error;

			/* Determine if the error is acceptable.  */
			if (test_error > ((float)0.1))
			{
				px5_neural_network_learn(xor_network, test_case_1, 2, test_case_1_expected, 1);
				still_learning = 1;
			}

			/* Compute the next test case.  */
			px5_neural_network_compute(xor_network, test_case_2, 2, results, 1);

			/* Calculate the error.  */
			if (results[0] > test_case_2_expected[0])
				test_error = results[0] - test_case_2_expected[0];
			else
				test_error = test_case_2_expected[0] - results[0];

			/* Keep track of the error for each pass.  */
			test_pass_error = test_pass_error + test_error;

			/* Determine if the error is acceptable.  */
			if (test_error > ((float)0.01))
			{
				px5_neural_network_learn(xor_network, test_case_2, 2, test_case_2_expected, 1);
				still_learning = 1;
			}

			/* Compute the next test case.  */
			px5_neural_network_compute(xor_network, test_case_3, 2, results, 1);

			/* Calculate the error.  */
			if (results[0] > test_case_3_expected[0])
				test_error = results[0] - test_case_3_expected[0];
			else
				test_error = test_case_3_expected[0] - results[0];

			/* Keep track of the error for each pass.  */
			test_pass_error = test_pass_error + test_error;

			/* Determine if the error is acceptable.  */
			if (test_error > ((float)0.01))
			{
				px5_neural_network_learn(xor_network, test_case_3, 2, test_case_3_expected, 1);
				still_learning = 1;
			}

			/* Compute the next test case.  */
			px5_neural_network_compute(xor_network, test_case_4, 2, results, 1);

			/* Calculate the error.  */
			if (results[0] > test_case_4_expected[0])
				test_error = results[0] - test_case_4_expected[0];
			else
				test_error = test_case_4_expected[0] - results[0];

			/* Keep track of the error for each pass.  */
			test_pass_error = test_pass_error + test_error;

			/* Determine if the error is acceptable.  */
			if (test_error > ((float)0.01))
			{
				px5_neural_network_learn(xor_network, test_case_4, 2, test_case_4_expected, 1);
				still_learning = 1;
			}

			if ((total_passes % 1000000) == 0)
			{

				px5_neural_network_print(xor_network);
				printf("Total Passes:  %d\n", total_passes);
				printf("Total error:   %f\n", test_pass_error);
			}
		} while (still_learning);
		/*percent_converge = percent_converge + 1;
		if (still_learning == 0)
		{
			complete_passes++;
			average_runs = average_runs + total_passes;
		}
		px5_neural_network_destroy(xor_network);
		px5_neural_network_create(&xor_network, xor_layer_specification);*/
	}
	/*printf("The network converged %.2f%% of the time\n", (percent_converge / ((float)10000)) * (float)100);
	printf("When converged it ran a average of %.1f times", ((float)average_runs) / ((float)complete_passes));*/

	px5_neural_network_print(xor_network);
	printf("\n\n***** Sucessful XOR training! ******\n\n");
	printf("Total Passes:  %d\n", total_passes);
	printf("Total error:   %f\n", test_pass_error);
	px5_neural_network_compute(xor_network, test_case_1, 2, results, 1);
	printf("XOR test case 1: 0 ^ 0 = %f\n", results[0]);
	px5_neural_network_compute(xor_network, test_case_2, 2, results, 1);
	printf("XOR test case 2: 0 ^ 1 = %f\n", results[0]);
	px5_neural_network_compute(xor_network, test_case_3, 2, results, 1);
	printf("XOR test case 3: 1 ^ 0 = %f\n", results[0]);
	px5_neural_network_compute(xor_network, test_case_4, 2, results, 1);
	printf("XOR test case 4: 1 ^ 1 = %f\n", results[0]);
}
