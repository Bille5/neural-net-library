#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "px5_neural_network.h"


int  	px5_neural_network_create(neural_network_handle *handle, int  *layer_specification)
{

int  			i, j, k;
int 			total_layers =   0;
int 			total_neurons =  0;
int  			total_connections =  0;
int         	previous_layer_neurons =  0;
unsigned long	memory_required;
unsigned char   *working_pointer;
neural_network  *neural_network_pointer;
layer 			*layer_pointer;
neuron 			*neuron_pointer;
connection 		*connection_pointer;
double          random_number;


    /* FIXME: Need to be able to specify the learning rate.  */

	/* Check for an invalid handle.  */
    if (handle == NULL)
		return(PX5_NEURAL_NETWORK_ERROR);

	/* Check for an invalid layer specification.  */
	if (layer_specification == NULL)
		return(PX5_NEURAL_NETWORK_ERROR);

	/* Loop through the layer specification to calculate number of layers, number of neurons, and number of connections.  */
	i =  0;
	while(layer_specification[i])	
	{

		/* Increment the number of layers.  */
		total_layers++;

		/* Compute the number of total neurons.  */
		total_neurons =  total_neurons + layer_specification[i];

		/* Compute the number of total connections.  */
		total_connections =  total_connections + (layer_specification[i] * previous_layer_neurons);

		/* Remember the previous layer neurons.  */
		previous_layer_neurons =  layer_specification[i];

		/* Move to the next layer in the specification.  */
		i++;
	}    

	/* Determine if there is a problem.  */
	if (total_layers <= 1)
		return(PX5_NEURAL_NETWORK_ERROR);

	/* Calculate the total amount of memory required.  */
	memory_required =   (sizeof(neural_network)) + (sizeof(layer)*total_layers) + (sizeof(neuron)*total_neurons) + (sizeof(connection)*total_connections);

	/* Allocate the memory required for the neural network.  */
	neural_network_pointer =  (neural_network *) calloc(memory_required, 1);

	/* Determine if there is an error.  */
	if (neural_network_pointer == NULL)
		return(PX5_NEURAL_NETWORK_ERROR);

	/* Now, we need to setup the neural network memory area.  */

	/* Populate the nework information at the top of the structure.  */
	neural_network_pointer -> id =   	           PX5_NEURAL_NETWORK_VALID;
	neural_network_pointer -> input_neurons =      layer_specification[0];
	neural_network_pointer -> output_neurons =     layer_specification[total_layers-1];
	neural_network_pointer -> total_layers =       total_layers;
	neural_network_pointer -> total_neurons =      total_neurons;
	neural_network_pointer -> total_connections =  total_connections;

	/* Setup a working pointer to the memory area.  */
	working_pointer =  (unsigned char *)  neural_network_pointer;

	/* Now calculate where the memory starts for the array of layers.  */
	working_pointer =  working_pointer + sizeof(neural_network);

	/* Setup the starting address of the list area.  */
	layer_pointer =  (layer *) working_pointer;

	/* Set up the layer list pointer in the neural network structure.  */
	neural_network_pointer -> layer_list =  layer_pointer;

	/* Now calcualte where the neuron area is, which immediately follows the layer list.  */
	working_pointer =  working_pointer + (sizeof(layer)*total_layers);

	/* Setup the starting address of the neuron area.  */
	neuron_pointer =  (neuron *) working_pointer;

	/* Set up the neuron area pointer in the neural network structure.  */
	neural_network_pointer -> neuron_area =  neuron_pointer;

	/* Now calculate where the connection area is, which immediately follows the neuron area.  */
	working_pointer =  working_pointer + (sizeof(neuron)*total_neurons);

	/* Setup the starting address of the connection area.  */
	connection_pointer =  (connection *)  working_pointer;

	/* Set up the connetion area pointer in the neural network structure.  */
	neural_network_pointer -> connection_area =  connection_pointer;

	/* Now, walk through each layer, initializing the each layer including all neurons and connections.  */
	for (i = 0; i < total_layers; i++)
	{

		/* Setup the neuron list for this layer.  */
		layer_pointer[i].neuron_list =  neuron_pointer;

		/* Remember the number of neurons at this layer.  */
		layer_pointer[i].neurons =  layer_specification[i];

		/* Initialize each neuron of this layer.  */
		for (j = 0; j < layer_specification[i]; j++)
		{

			/* Setup the layer and neuron number just for future reference.  */
			neuron_pointer[j].layer =          i;
			neuron_pointer[j].neuron_number =  j;
			neuron_pointer[j].bias =           PX5_NEURAL_NETWORK_BIAS;

			/* Determine if this is not the input layer.  */
			if (i > 0)
			{

				/* Not the input layer.  */

				/* Calculate the number of connections, which is the neurons in the previous layer.  */
				neuron_pointer[j].connections_to_previous_layer =  layer_specification[i-1];

				/* Setup the connection list for this neuron.  */
				neuron_pointer[j].connection_list =  connection_pointer;

				/* Loop through the connections, settup up each connection weight.  */
				for (k = 0; k < neuron_pointer[j].connections_to_previous_layer; k++)
				{
					//srand(time(NULL));
					/* Set the connection weight to random value.  */
					//connection_pointer[k].weight =  ((float) rand() / (float) RAND_MAX) - 0.5;
					connection_pointer[k].weight = ((float)arc4random() / (float)PX5_UINT_MAX) - 0.5;
				}

				/* Move the connection pointer to the next next free connection.  */
				connection_pointer =  connection_pointer + neuron_pointer[j].connections_to_previous_layer;
			}
		}

		/* Move the neuron pointer to the next free neuron.  */
		neuron_pointer =  neuron_pointer + layer_pointer[i].neurons;
	}

	/* Setup the neural network learning rate.  */
	neural_network_pointer->learning_rate  =  PX5_NEURAL_NETWORK_LEARNING_RATE;


	/* Return the neural network handle.  */
	*handle =  neural_network_pointer;

	/* Return successful status.  */
	return(PX5_NEURAL_NETWORK_SUCCESS);
}


int     px5_neural_network_compute(neural_network_handle handle, float *input_list, int number_of_inputs, float *output_list, int  nubmer_of_outputs)
{

neural_network 		*neural_network_pointer;
int                 i, j, k;
int                 total_layers;
layer               *layer_list;
neuron  		    *neuron_list;
connection          *connection_list;
double              summation, activation_value;


	/* Pickup the neural network pointer.  */
	neural_network_pointer =  handle;

	/* Check to see if it is valid.  */
	if (neural_network_pointer == NULL)
		return(PX5_NEURAL_NETWORK_ERROR);

	if (neural_network_pointer -> id != PX5_NEURAL_NETWORK_VALID)
		return(PX5_NEURAL_NETWORK_ERROR);

	/* Check for the correct number of inputs.  */
	if (neural_network_pointer -> input_neurons != number_of_inputs)
		return(PX5_NEURAL_NETWORK_ERROR);

	/* Check for the correct number of outputs.  */
	if (neural_network_pointer -> output_neurons != nubmer_of_outputs)
		return(PX5_NEURAL_NETWORK_ERROR);

	/* Setup the layer list.  */
	layer_list =  neural_network_pointer -> layer_list;

	/* Pickup the total number of layers.  */
	total_layers =  neural_network_pointer -> total_layers;

	/* Setup the neuron list for the input layer.  */
	neuron_list =  layer_list[0].neuron_list;

	/* Move the inputs into the input layer values.  */
	for (j = 0; j < layer_list[0].neurons; j++)
	{

		/* Copy the user input into the input layer.  */
		neuron_list[j].value =  input_list[j];
	}

	/* Calculate the outputs!  */

	/* Loop through each layer starting at layer 1.  */
	for (i = 1; i < total_layers; i++)
	{

		/* Setup the neuron list pointer.  */
		neuron_list =  layer_list[i].neuron_list;

		/* Loop through the neurons in this layer.  */
		for (j = 0; j < layer_list[i].neurons; j++)
		{

			/* Initialize the sumation to 0.  */
			summation = 0.0;

			/* Setup the connection list for this neuron.  */
			connection_list =  neuron_list[j].connection_list;

			/* Loop through the connection for this neuron, computing the value.  */
			for (k = 0; k < neuron_list[j].connections_to_previous_layer; k++)
			{

				/* Sum the weights * previous neurons value  */
				summation =  summation + (layer_list[i-1].neuron_list[k].value * connection_list[k].weight);
			}

			/* Now add the bias to the summation.  */
			summation = summation + neuron_list[j].bias;

			/* FIXME:  Other activation function support?  Maybe allow user to decide?  */

			/* Now compute the output of this neuron using simple Sigmoid activation function: 1/(1 + e(-summation)).  */
			activation_value = ((float)1.0) / (((float)1.0) + exp(-summation));

			/* Clear this neuron's value.  */
			neuron_list[j].value =  (float) activation_value;
		}
	}

	/* FIXME: If NULL is supplied for output, don't copy the output.  */

	/* Setup the neuron list the output layer.  */
	neuron_list =  layer_list[total_layers-1].neuron_list;

	/* Copy the ouptput values into the caller's space.  */
	for (j = 0; j < neural_network_pointer -> output_neurons; j++)
	{

		/* Copy each neuron value into the output.  */
		output_list[j] =  neuron_list[j].value;
	}

	/* Return success!  */
	return(PX5_NEURAL_NETWORK_SUCCESS);
}


int     px5_neural_network_learn(neural_network_handle handle, float *input_list, int number_of_inputs, float *expected_output_list, int  expected_number_of_outputs)
{

neural_network 		*neural_network_pointer;
int                 i, j, k;
int                 total_layers;
layer*              layer_list;
neuron*             neuron_list;
connection*         connection_list;
double              delta, error, weight;
double              weight_delta_summation;
double              weight_adjust;


    /* FIXME: this function should all the computation function to make it easier for the user - less API calls.  */

    /* FIXME: Need to add batch logic and a way to specific the batch size per neural network.  */

	/* Pickup the neural network pointer.  */
	neural_network_pointer =  handle;

	/* Check to see if it is valid.  */
	if (neural_network_pointer == NULL)
		return(PX5_NEURAL_NETWORK_ERROR);

	if (neural_network_pointer -> id != PX5_NEURAL_NETWORK_VALID)
		return(PX5_NEURAL_NETWORK_ERROR);

	/* Setup the layer list.  */
	layer_list = neural_network_pointer->layer_list;

	/* Pickup the total number of layers.  */
	total_layers = neural_network_pointer->total_layers;

	/* Setup the layer index.  */
	i = total_layers - 1;

	/* Setup the neuron list for the output layer.  */
	neuron_list = layer_list[i].neuron_list;

	/* Make sure there are enough output neurons.  */
	if (layer_list[i].neurons < expected_number_of_outputs)
		return(PX5_NEURAL_NETWORK_ERROR);

	/* Calculate the output layer delta.  */
	for (j = 0; j < layer_list[i].neurons; j++)
	{ 

		/* Output layer delta computation is:  (value * (1-value)) * (expected_value - value)  */

		/* First cacluate the error.  */
		error = (double) (expected_output_list[j] - neuron_list[j].value);

		/* Calculate the delta.  */
		delta = (double) neuron_list[j].value;

		/* First part of delta for output layer.  */
		delta = delta * (((double)1.0) - delta);

		/* Final computation of the delta for the output layer.  */
		delta = delta * error;

		neuron_list[j].delta =  (float) delta;
	}

	/* Calculate the error for each layer.  */

	/* Move to the layer immediately before the output layer.  */
	i--;
	while (i)
	{

		/* Calculate the non-output layer neuron delta.  */
		for (j = 0; j < layer_list[i].neurons; j++)
		{

			/* Non-output layer delta computation is:  (value * (1-value)) * sumation (weights * delta of connected node)  */

			/* First, cacluate the summation of the weights * delta of connected node.  */
			weight_delta_summation = (double) 0.0;
			for (k = 0; k < layer_list[i + 1].neurons; k++)
			{
				connection_list = layer_list[i + 1].neuron_list[k].connection_list;
				weight_delta_summation = weight_delta_summation + (((double)layer_list[i + 1].neuron_list[k].delta) * ((double)connection_list[j].weight));
			}

			/* Calculate the delta.  */
			delta = (double) layer_list[i].neuron_list[j].value;

			/* First part of delta for output layer.  */
			delta = delta * (((double)1.0) - delta);

			/* Final computation of the delta for the output layer.  */
			delta = delta * weight_delta_summation;

			layer_list[i].neuron_list[j].delta = (float)delta;
		}

		/* Move to the previous lower layer.  */
		i--;
	}

	/* FIXME:  Need to adjust bias in the learning.  */

	/* Now that we have all the deltas computed, we need adjust the weights accordingly.  */
	for (i = 1; i < total_layers; i++)
	{

		/* Walk through all neurons of this layer.  */
		for (j = 0; j < layer_list[i].neurons; j++)
		{

			/* Adjust the weights for all connections to this neuron.  */
			for (k = 0; k < layer_list[i-1].neurons; k++)
			{

				connection_list = layer_list[i].neuron_list[j].connection_list;

				/* Pickup the current weight.  */
				weight = (double) connection_list[k].weight;

				/* Calculate the amount we are going to adjust the weight.  FIXME: learning rate goes here! */
				weight_adjust =  ((((double)layer_list[i-1].neuron_list[k].value) * ((double) layer_list[i].neuron_list[j].delta)));

				/* Apply the learning rate.  */
				weight_adjust = weight_adjust * ((double)neural_network_pointer->learning_rate);

				/* Calculate the new weight.  */
				weight = weight + weight_adjust;

				/* Store the weight back in the connection.  */
				connection_list[k].weight = (float) weight;
			}
		}
	}

	return(PX5_NEURAL_NETWORK_SUCCESS);
}


int     px5_neural_network_destroy(neural_network_handle  handle)
{

neural_network 		*neural_network_pointer;


	/* Pickup the neural network pointer.  */
	neural_network_pointer =  handle;

	/* Check to see if it is valid.  */
	if (neural_network_pointer == NULL)
		return(PX5_NEURAL_NETWORK_ERROR);

	if (neural_network_pointer -> id != PX5_NEURAL_NETWORK_VALID)
		return(PX5_NEURAL_NETWORK_ERROR);

	/* Free the memory for the neural network.  */
	free((void *) neural_network_pointer);

	return(PX5_NEURAL_NETWORK_SUCCESS);
}


int     px5_neural_network_neuron_print(neural_network_handle handle, int  layer_requested, int  neuron_requested)
{

	neural_network*     neural_network_pointer;
	int                 i;
	layer*              layer_list;
	neuron*             neuron_pointer;
	neuron*             next_layer_neuron_pointer;
	connection*         connection_list;


	/* Pickup the neural network pointer.  */
	neural_network_pointer = handle;

	/* Check to see if it is valid.  */
	if (neural_network_pointer == NULL)
		return(PX5_NEURAL_NETWORK_ERROR);

	if (neural_network_pointer->id != PX5_NEURAL_NETWORK_VALID)
		return(PX5_NEURAL_NETWORK_ERROR);

	/* Check for a correct layer.  */
	if (layer_requested >= neural_network_pointer->total_layers)
		return(PX5_NEURAL_NETWORK_ERROR);

	/* Setup the layer list.  */
	layer_list = neural_network_pointer->layer_list;

	/* Check for a corrent neuron.  */
	if (neuron_requested >= layer_list[layer_requested].neurons)
		return(PX5_NEURAL_NETWORK_ERROR);

	/* When we get here everything is good, print out the neuron!  */
	neuron_pointer = &layer_list[layer_requested].neuron_list[neuron_requested];

	if (neuron_pointer->layer != layer_requested)
		printf("Invalid Layer!!!\n");
	if (neuron_pointer->neuron_number != neuron_requested)
		printf("Invalid Neuron!!!!\n");

	/* Print out the neuron!  */
	if (layer_requested == 0)
	{

		/* Input layer.  */
		printf("**************** Input Layer 0 - Neuron: %d ****************\n", neuron_requested);
		printf("Input Value:  %f\n", neuron_pointer -> value);

		/* Print out connections to next layers.  */
		printf("Connections to next layer:\n");
		next_layer_neuron_pointer = layer_list[layer_requested + 1].neuron_list;
		for (i = 0; i < (layer_list[layer_requested + 1].neurons); i++)
		{
			connection_list = next_layer_neuron_pointer[i].connection_list;
			printf("    Connection to layer %d, neuron %d:  weight: %f\n", (layer_requested + 1), i, connection_list[neuron_requested].weight);
		}
	}
	else if (layer_requested < (neural_network_pointer->total_layers - 1))
	{

		/* Hidden layer.  */
		printf("**************** Hidden Layer %d - Neuron: %d ****************\n", layer_requested, neuron_requested);
		printf("Value:  %f, Bias: %f, Delta: %f\n", neuron_pointer->value, neuron_pointer->bias, neuron_pointer->delta);

		/* Print out connections to next layers.  */
		printf("Connections to next layer:\n");
		next_layer_neuron_pointer = layer_list[layer_requested + 1].neuron_list;
		for (i = 0; i < (layer_list[layer_requested + 1].neurons); i++)
		{
			connection_list = next_layer_neuron_pointer[i].connection_list;
			printf("    Connection to layer %d, neuron %d:  weight: %f\n", (layer_requested + 1), i, connection_list[neuron_requested].weight);
		}
	}
	else
	{

		/* Output layer.  */
		printf("**************** Output Layer %d - Neuron: %d ****************\n", layer_requested, neuron_requested);
		printf("Value:  %f, Bias: %f, Delta: %f\n", neuron_pointer->value, neuron_pointer->bias, neuron_pointer->delta);
	}

	return(PX5_NEURAL_NETWORK_SUCCESS);
}

int     px5_neural_network_print(neural_network_handle handle)
{

	neural_network* neural_network_pointer;
	int             i, j;


	/* Pickup the neural network pointer.  */
	neural_network_pointer = handle;

	/* Check to see if it is valid.  */
	if (neural_network_pointer == NULL)
		return(PX5_NEURAL_NETWORK_ERROR);

	if (neural_network_pointer->id != PX5_NEURAL_NETWORK_VALID)
		return(PX5_NEURAL_NETWORK_ERROR);

	/* Print banner.  */
	printf("*****************  NEURAL NETWORK PRINT *******************\n\n");

	/* Loop through the entire network to print out all neurons and connections.  */
	for (i = 0; i < neural_network_pointer->total_layers; i++)
	{

		/* Pickup the layer list.  */
		for (j = 0; j < neural_network_pointer->layer_list[i].neurons; j++)
		{

			/* Print the neuron out!  */\
				px5_neural_network_neuron_print(handle, i, j);
		}
	}
	printf("\n\n");

	return(PX5_NEURAL_NETWORK_SUCCESS);
}

