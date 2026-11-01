
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

/* Preference choices */
#define NO_ACTIVATION 0
#define SIGMOID 1
#define RELU 2
#define LEAKY_RELU 3
#define ELU 4
#define SELU 5
#define GELU 6
#define TAHN 7
#define SWISH 8
#define SOFTSIGN 9
#define HARD_SIGMOID 10
#define LOG_SIGMOID 11
#define SOFTPLUS 12
#define BINARY 13
#define MISH 14
#define ARCTAN 15
#define SINUSOID 16
#define GAUSSIAN 17
#define BENT_IDENTITY 18

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
	float			mean_of_weights;
	float           variance_of_weights;
} connection;


typedef struct neuron_struct
{
  	int   			layer;
  	int   			neuron_number;
  	float 			value;
  	float 			bias;
	float			mean_of_biases;
	float			variance_of_biases;
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
	int             iterations;

  	layer           *layer_list;
  	neuron          *neuron_area;
  	connection      *connection_area;
} neural_network;

/* Attribute structure */
typedef struct network_attributes_struct
{
	int    activation_function;
	int    optimization;
	int    regularization;
	float  learning_rate;
	int    batch_size;
	int    weight_initialization;
	int    bias_initialization;
} network_attributes;


/* Define handle to neural network, which is how the APIs will reference it.  */

typedef neural_network * neural_network_handle;


/* Define function prototypes for the neural network.  */

int  	px5_neural_network_create(neural_network_handle *handle, int  *layer_specification, network_attributes *attributes);
int     px5_neural_network_compute(neural_network_handle handle, float *input_list, int number_of_inputs, float *output_list, int  number_of_outputs);
int     px5_neural_network_learn(neural_network_handle handle, float *input_list, int number_of_inputs, float *expected_output_list, int  expected_number_of_outputs);
int     px5_neural_network_destroy(neural_network_handle  handle);
int     px5_neural_network_neuron_print(neural_network_handle handle, int  layer_requested, int  neuron_requested);
int     px5_neural_network_print(neural_network_handle handle);
double  px5_activation_function(neural_network_handle handle, int layer, double number);
double   px5_activation_derivative(neural_network_handle handle, int layer, double number);
double px5_weight_initialization(neural_network_handle handle, double number);
double px5_weight_initialization(neural_network_handle handle, double number);
float   px5_random_uniform_number(float x, float y);
int  px5_neural_network_attributes_create(network_attributes *attributes);
int  px5_neural_network_attribute_activation_change(network_attributes *attributes, int  new_activation_function, int new_optimization, int new_regularization, float new_learning_rate, int new_batch_size, int new_bias_initialization, int new_weight_inizialization);



#endif  /* PX5_NEURAL_NETWORK_H  */