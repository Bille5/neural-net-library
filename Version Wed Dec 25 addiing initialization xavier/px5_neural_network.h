
#ifndef PX5_NEURAL_NETWORK_H
#define PX5_NEURAL_NETWORK_H


/* Define constants.  */

#define PX5_NEURAL_NETWORK_VALID           0x50584E4EUL
#define PX5_NEURAL_NETWORK_ERROR           -1
#define PX5_NEURAL_NETWORK_SUCCESS         0
#define PX5_NEURAL_NETWORK_BIAS            ((float) 0.0)
#define PX5_NEURAL_NETWORK_LEARNING_RATE   ((float) 1)
#define PX5_NULL                           ((void *) 0)
#define PX5_UINT_MAX                       ((float)4294967295)


/* Define data structures.  */

/* The high-level overview of the neural network data structure is as follows:

	[ Overview 	   ]    (typedef neural_network)        
	[ Layer Area  	   ]   	(typedef layer (array))  
	[ Neuron Area 	   ] 	(typedef neuron (array))
	[ Connection Area  ]	(typedef connection (array))


	The memory for all of these areas will be allocated dynamically via malloc.

*/

typedef struct connection_struct
{
  	float   		weight;
} connection;


typedef struct neuron_struct
{
  	int   			layer;
  	int   			neuron_number;
  	float 			value;
  	float 			bias;
  	float 			delta;
  	int        		connections_to_previous_layer;
  	connection 		*connection_list;
} neuron;


typedef struct layer_struct
{
  	int     		neurons;
  	neuron 			*neuron_list;
} layer;


typedef struct neural_network_struct
{
	unsigned long 	id;
	int             input_neurons;
	int             output_neurons;
  	int             total_layers;
	int             total_neurons;
	int             total_connections;
        float           learning_rate;
  	layer           *layer_list;
  	neuron          *neuron_area;
  	connection      *connection_area;
} neural_network;


/* Define handle to neural network, which is how the APIs will reference it.  */

typedef neural_network * neural_network_handle;


/* Define function prototypes for the neural network.  */

int  	px5_neural_network_create(neural_network_handle *handle, int  *layer_specification);
int     px5_neural_network_compute(neural_network_handle handle, float *input_list, int number_of_inputs, float *output_list, int  number_of_outputs);
int     px5_neural_network_learn(neural_network_handle handle, float *input_list, int number_of_inputs, float *expected_output_list, int  expected_number_of_outputs);
int     px5_neural_network_destroy(neural_network_handle  handle);
int     px5_neural_network_neuron_print(neural_network_handle handle, int  layer_requested, int  neuron_requested);
int     px5_neural_network_print(neural_network_handle handle);
float   px5_random_uniform_number(float x, float y);


#endif  /* PX5_NEURAL_NETWORK_H  */