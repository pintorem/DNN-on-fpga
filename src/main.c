#include <stdio.h>
#include <stdlib.h>
#include "platform.h"
#include "xil_printf.h"
#include "xuartps.h"
#include "weights.h"
#include <xtime_l.h>
#include <time.h>
#include <math.h>
#include <arm_neon.h>


#define TIME_COUNT 1
#define OPTIMIZED 2 // OPTIMIZED BELONGS TO {1,2,3,4}. 1 = baseline, 2 = timecount, 3 = optimized, 4 = intrinsics


#if OPTIMIZED==1 // Q8.8 Mode
	#define QF 8
	typedef short int DATA;
	typedef short int NN_PARAM;
	typedef short int IMAGE_DATA_T;

	NN_PARAM gemm0_bias[N_BIAS0] = { bias0_q88 };
	NN_PARAM gemm0_weights[N_WEIGHTS0] = { weights0_q88 };
	NN_PARAM gemm1_bias[N_BIAS1] = { bias1_q88 };
	NN_PARAM gemm1_weights[N_WEIGHTS1] = { weights1_q88 };
#elif OPTIMIZED==2 // Q0.8 img and Q1.7 net params
	#define QF 7
	typedef short int DATA;
	typedef int8_t NN_PARAM;
	typedef u8 IMAGE_DATA_T;

	NN_PARAM gemm0_bias[N_BIAS0] = {0};
	NN_PARAM gemm0_weights[N_WEIGHTS0] = {0};
	NN_PARAM gemm1_bias[N_BIAS1] = {0};
	NN_PARAM gemm1_weights[N_WEIGHTS1] = {0};
#elif OPTIMIZED==3 // Q0.8 img and Q1.7 net params + ARM features
	#define QF 7
	typedef short int DATA;
	typedef int8_t NN_PARAM;
	typedef u8 IMAGE_DATA_T;

	NN_PARAM gemm0_bias[N_BIAS0] = {0};
	NN_PARAM gemm0_weights[N_WEIGHTS0] = {0};
	NN_PARAM gemm1_bias[N_BIAS1] = {0};
	NN_PARAM gemm1_weights[N_WEIGHTS1] = {0};
#endif


// Img defines
#define IMG_DIM_1 28
#define IMG_DIM_2 28
#define IMG_SIZE IMG_DIM_1 * IMG_DIM_2
#define SIZEWA 10

#define FIXED2FLOAT(a, qf) (((float) (a)) / (1<<qf))
#define FLOAT2FIXED(a, qf) ((short int) round((a) * (1<<qf)))

#define _MAX_ (1 << (sizeof(DATA)*8-1))-1
#define _MIN_ -(_MAX_+1)

// Time utils
#define CLOCK_PERIOD (1/XPAR_CPU_CORTEXA9_0_CPU_CLK_FREQ_HZ)
XTime time_start, time_end, overall_time, overall_start_time;

//======================================================== FUNCTION PROTOTYPES
void FC_forward(DATA* input, DATA* output, int in_s, int out_s, NN_PARAM* weights, NN_PARAM* bias, int qf);
void FC_forward_layer0(IMAGE_DATA_T* input, DATA* output, int in_s, int out_s, NN_PARAM* weights, NN_PARAM* bias, int qf);
static inline long long int saturate(long long int mac);
static inline void relu_forward(DATA* input, DATA* output, int size);
int resultsProcessing(DATA* results, int size);
XUartPs_Config* setup_uart();
void read_from_uart(u32 BaseAddress, DATA * ptr, int size);
void readfromUART_8(u32 BaseAddress, IMAGE_DATA_T buffer_read[] , int numBytes);

// Function useful to load dynamically network parameters from the uart
void read_network_parameters_uart(XUartPs_Config* config, NN_PARAM* gemm0_bias, int dim_gemm0_bias, NN_PARAM* gemm0_weights, int dim_gemm0_weights,
        												  NN_PARAM* gemm1_bias, int dim_gemm1_bias, NN_PARAM* gemm1_weights, int dim_gemm1_weights);

long long int dot_product_intrinsic(DATA* vec1, DATA* vec2, int n);
void FC_forward_intrinsics(DATA * input, DATA * output, int in_s, int out_s, NN_PARAM* weights, NN_PARAM* bias, int qf);
int DNN_forward(IMAGE_DATA_T *image, int img_size, void (*FC_forward)(DATA*, DATA*, int, int, NN_PARAM*, NN_PARAM*, int));
//======================================================== END FUNCTION PROTOTYPES


int main() {
	init_platform();
	XUartPs_Config* uart_config = setup_uart();

	xil_printf("\n\n\n\n\r========================== Welcome to the DNN Classifier");

	//=================================== Prepare function pointers based which kind of execution is desired
	IMAGE_DATA_T image[IMG_SIZE];
	int (*forward)(IMAGE_DATA_T*, int, void (*FC_forward)(DATA*, DATA*, int, int, NN_PARAM*, NN_PARAM*, int));
	void (*fc_forward)(DATA*, DATA*, int, int, NN_PARAM*, NN_PARAM*, int);
	void (*read_image)(u32, IMAGE_DATA_T*, int);
	void (*read_network_parameters)(XUartPs_Config*, NN_PARAM*, int, NN_PARAM*, int,NN_PARAM* , int, NN_PARAM*, int);

#if OPTIMIZED==1 // Q8.8
	read_image = read_from_uart;
	forward = DNN_forward;
	fc_forward = FC_forward;
	read_network_parameters = NULL;
#elif OPTIMIZED==2 // Q0.8 img and Q1.7 net params
	read_image = readfromUART_8;
	forward = DNN_forward;
	fc_forward = FC_forward;
	read_network_parameters = read_network_parameters_uart;
#elif OPTIMIZED==3 // Q0.8 img and Q1.7 net params + ARM features
	read_image = readfromUART_8;
	forward = DNN_forward;
	fc_forward = FC_forward_intrinsics;
	read_network_parameters = read_network_parameters_uart;
#endif

	//=================================== Read the network configuration through UART
	if(read_network_parameters != NULL)
			read_network_parameters(uart_config, gemm0_bias, N_BIAS0, gemm0_weights, N_WEIGHTS0,
			  	  	  	  	  	  	  	   	     gemm1_bias, N_BIAS1, gemm1_weights, N_WEIGHTS1);

	//=================================== Start reading images and use the neural net to make classifications
	while (1) {
		xil_printf("\n\rWaiting for the image...");

		// Read image from the uart
		read_image(uart_config->BaseAddress, image, IMG_SIZE);

		// Process the image with the DNN according
		int classification = forward(image, IMG_SIZE, fc_forward);


		// send the result in output to the UART
		xil_printf("\r");
		xil_printf("\n\rClassification DNN = %x", classification);

		if(TIME_COUNT){
			XTime overall_time;
			XTime_GetTime(&overall_time);
			printf("\n\rCycles: %llu - us %.2f us.", overall_time-overall_start_time, 1.0 * (overall_time - overall_start_time) / (COUNTS_PER_SECOND/1000000));
		}
	}
	//=================================== End of processing images

	xil_printf("\n\r========================== BYE!");

	cleanup_platform();
	return 0;
}

XUartPs_Config* setup_uart() {
	//UART setup
	XUartPs Uart_1_PS;
	u16 DeviceId_1 = XPAR_PS7_UART_1_DEVICE_ID;
	int Status_1;
	XUartPs_Config *Config_1;
	Config_1 = XUartPs_LookupConfig(DeviceId_1);
	if (NULL == Config_1) {
		return XST_FAILURE;
	}
	/*the default configuration is stored in Config and it can be used to initialize the controller */
	Status_1 = XUartPs_CfgInitialize(&Uart_1_PS, Config_1,
			Config_1->BaseAddress);
	if (Status_1 != XST_SUCCESS) {
		return XST_FAILURE;
	}
	// Set the BAUD rate
	u32 BaudRate = (u32) 115200;
	Status_1 = XUartPs_SetBaudRate(&Uart_1_PS, BaudRate);
	if (Status_1 != (s32) XST_SUCCESS) {
		return XST_FAILURE;
	}
	//END UART SETUP
	return Config_1;
}

// Functions useful to read 16-bit images from uart
void read_from_uart(u32 BaseAddress, DATA* ptr, int size){
	int i;
	u8 data1, data2;

	for (i=0; i<size; i++) {
		if (TIME_COUNT && i==1) XTime_GetTime(&overall_start_time);
		data1 = XUartPs_RecvByte(BaseAddress);
		data2 = XUartPs_RecvByte(BaseAddress);
		*(ptr+i) = (data2 << 8) + data1;
	}

	if(TIME_COUNT){
		XTime end_read;
		XTime_GetTime(&end_read);
		printf("\n\rCycles: %llu - us %.2f us.", end_read-overall_start_time, 1.0 * (end_read - overall_start_time) / (COUNTS_PER_SECOND/1000000));
	}
}

void readfromUART_8(u32 BaseAddress, IMAGE_DATA_T buffer_read[] , int numBytes){
	int i;
	IMAGE_DATA_T data1;

	for (i=0; i<numBytes; i++) {
		if (TIME_COUNT && i==1) XTime_GetTime(&overall_start_time);
		data1 = XUartPs_RecvByte(BaseAddress);
		buffer_read[i] = data1;
	}

	if(TIME_COUNT){
		XTime end_read;
		XTime_GetTime(&end_read);
		printf("\n\rCycles: %llu - us %.2f us.", end_read-overall_start_time, 1.0 * (end_read - overall_start_time) / (COUNTS_PER_SECOND/1000000));
	}
}

void read_net_parameters_q17(u32 BaseAddress, NN_PARAM *ptr, int size){
	int i;
	u8 data1, data2;

	for (i=0; i<size; i++) {
		if (i==1)
			time_start = Xil_In32(GLOBAL_TMR_BASEADDR + GTIMER_COUNTER_LOWER_OFFSET);

		data1 = XUartPs_RecvByte(BaseAddress);
		data2 = XUartPs_RecvByte(BaseAddress);

		// Convert to q1.7
		ptr[i] = (data2 == 0xFF) ? ((0b1 << 7) + (data1 >> 1)) : ((0b0 << 7) + (data1 >> 1));
	}
}

void read_network_parameters_uart(XUartPs_Config* config, NN_PARAM* gemm0_bias, int dim_gemm0_bias, NN_PARAM* gemm0_weights, int dim_gemm0_weights,
                                                          NN_PARAM* gemm1_bias, int dim_gemm1_bias, NN_PARAM* gemm1_weights, int dim_gemm1_weights){

    xil_printf("\n\rNeural network configuration procedure");
    // layer 0
	xil_printf("\n\rSend me biases for gemm0 \n");
	read_net_parameters_q17(config->BaseAddress, gemm0_bias, dim_gemm0_bias);
	xil_printf("\n\rThanks for biases for gemm0!\n");
	xil_printf("\n\rSend me weights for gemm0\n");
	read_net_parameters_q17(config->BaseAddress, gemm0_weights, dim_gemm0_weights);
	xil_printf("\n\rThanks for weights for gemm0!\n");

	// layer 1
	xil_printf("\n\rSend me biases for gemm1\n");
	read_net_parameters_q17(config->BaseAddress, gemm1_bias, dim_gemm1_bias);
	xil_printf("\n\rThanks for biases for gemm1!\n");
	xil_printf("\n\rSend me weights for gemm1\n");
	read_net_parameters_q17(config->BaseAddress, gemm1_weights, dim_gemm1_weights);
	xil_printf("\n\rThanks for weights for gemm1!\n");
}

int DNN_forward(IMAGE_DATA_T *image, int img_size, void (*FC_forward)(DATA*, DATA*, int, int, NN_PARAM*, NN_PARAM*, int)) {
	int output_data = -1; // -1 means that there is an error in the classification
	XTime t_start_linear,t_end_linear;
	XTime t_start_relu,t_end_relu;

	/*
	 * Since each neuron will output only one value, we know that each neuron has one bias,
	 * so we can use the information about the number of biases to get the dimension of the
	 * output tensor at each layer
	 */

	// Forward layer 0
	DATA out_gemm0[N_BIAS0];  // prepare the output of the forward layer
	DATA input_gemm1[N_BIAS0]; // prepare the output of the relu (that will go be fed in the next layer)

	XTime_GetTime(&t_start_linear);
#if OPTIMIZED==2 || OPTIMIZED==3
	FC_forward_layer0(image, out_gemm0, IMG_SIZE, N_BIAS0, gemm0_weights, gemm0_bias, QF);
#else
	FC_forward(image, out_gemm0, IMG_SIZE, N_BIAS0, gemm0_weights, gemm0_bias, QF);
#endif
	XTime_GetTime(&t_end_linear);
	printf("\n\rF0: Cycles: %llu - us %.2f us.", t_end_linear-t_start_linear, 1.0 * (t_end_linear-t_start_linear) / (COUNTS_PER_SECOND/1000000));

	XTime_GetTime(&t_start_relu);
	relu_forward(out_gemm0, input_gemm1, N_BIAS0);
	XTime_GetTime(&t_end_relu);

	printf("\n\rR0: Cycles: %llu - us %.2f us.", t_end_relu-t_start_relu, 1.0 * (t_end_relu-t_start_relu) / (COUNTS_PER_SECOND/1000000));

	// End forward layer 0

	// Forward layer 1
	DATA out_gemm1[N_BIAS1];
	XTime_GetTime(&t_start_linear);
	FC_forward(input_gemm1, out_gemm1, N_BIAS0, N_BIAS1, gemm1_weights, gemm1_bias, QF);
	XTime_GetTime(&t_end_linear);
	printf("\n\rF1: Cycles: %llu - us %.2f us.", t_end_linear-t_start_linear, 1.0 * (t_end_linear-t_start_linear) / (COUNTS_PER_SECOND/1000000));
	// End forward layer 1

	// Classification
	XTime_GetTime(&t_start_linear);
	output_data = resultsProcessing(out_gemm1, N_BIAS1);
	XTime_GetTime(&t_end_linear);
	printf("\n\rClassif.: Cycles: %llu - us %.2f us.", t_end_linear-t_start_linear, 1.0 * (t_end_linear-t_start_linear) / (COUNTS_PER_SECOND/1000000));

	return output_data;
}

void FC_forward_layer0(IMAGE_DATA_T * input, DATA * output, int in_s, int out_s, NN_PARAM * weights, NN_PARAM* bias, int qf) {
	int hkern = 0;
	int wkern = 0;
	long long int mac = 0;
	u8 current = 0;

	for (hkern = 0; hkern < out_s; hkern++) {
		mac = ((long long int)bias[hkern]) << qf;
		for (wkern = 0; wkern < in_s; wkern++) {
			current = input[wkern];
			mac += current * weights[hkern*in_s + wkern];
		}
		output[hkern] = (DATA)saturate(mac >> qf);
	}
}

void FC_forward(DATA * input, DATA * output, int in_s, int out_s, NN_PARAM * weights, NN_PARAM* bias, int qf) {
	int hkern = 0;
	int wkern = 0;
	long long int mac = 0;
	DATA current = 0;

	for (hkern = 0; hkern < out_s; hkern++) {
		mac = ((long long int)bias[hkern]) << qf;
		for (wkern = 0; wkern < in_s; wkern++) {
			current = input[wkern];
			mac += current * weights[hkern*in_s + wkern];
		}
		output[hkern] = (DATA)saturate(mac >> qf);
	}
}

long long int dot_product_intrinsic(DATA * vec1, DATA * vec2, int n) {
	int16x4_t vec1_q, vec2_q;
	int32x4_t sum_q = {0, 0, 0, 0};
	int32x2_t tmp[2];
	long long int result;
	for( int i=0; i<( n & ~3); i+=4 ) {
		vec1_q = vld1_s16(&vec1[i]);
		vec2_q = vld1_s16(&vec2[i]);
		sum_q += vmull_s16(vec1_q, vec2_q );
	}
	tmp[0] = vget_high_s32(sum_q);
	tmp[1] = vget_low_s32 (sum_q);
	tmp[0] = vpadd_s32(tmp[0], tmp[1]);
	tmp[0] = vpadd_s32(tmp[0], tmp[0]);
	result = vget_lane_s32(tmp[0], 0);
	return result;
}

void FC_forward_intrinsics(DATA * input, DATA * output, int in_s, int out_s, NN_PARAM* weights, NN_PARAM* bias, int qf){
	int hkern = 0;
	long long int mac = 0;

	for (hkern = 0; hkern < out_s; hkern++) {
		mac = ((long long int)bias[hkern]) << qf;
		mac += dot_product_intrinsic(input, (DATA *)(weights+(hkern*in_s)), in_s);
		output[hkern] = (DATA)saturate(mac >> qf);
	}
}

static inline long long int saturate(long long int mac) {

	if (mac > _MAX_) {
		//printf("[WARNING] Saturation.mac: %lld -> %llx _MAX_: %d  _MIN_: %d  res: %d\n",mac, mac, _MAX_, _MIN_, _MAX_);
		return _MAX_;
	}

	if (mac < _MIN_) {
		//printf("[WARNING] Saturation. mac: %lld -> %llx _MAX_: %d  _MIN_: %d  res: %d\n",mac, mac, _MAX_, _MIN_, _MIN_);
		return _MIN_;
	}

	//printf("mac: %lld -> %llx _MAX_: %lld  _MIN_: %lld  res: %lld\n", mac, mac, _MAX_, _MIN_, mac);
	return mac;

}

static inline void relu_forward(DATA* input, DATA* output, int size) {
	int i = 0;
	for (i = 0; i < size; i++) {
		DATA v = input[i];
		v = v > 0 ? v : 0;
		output[i] = v;
	}
}

int resultsProcessing(DATA* results, int size) {
	int size_wa = SIZEWA;
	float r[SIZEWA];
	int c[SIZEWA];
	float results_float[SIZEWA];
	float sum = 0.0;
	DATA max = 0;
	int max_i;
	for (int i = 0; i < size_wa; i++) {
		results_float[i] = FIXED2FLOAT(results[i], 8);
		int n;
		if (results[i] > 0)
			n = results[i];
		else
			n = -results[i];
		if (n > max) {
			max = n;
			max_i = i;
		}
	}
	for (int i = 0; i < size_wa; i++)
		sum += exp(results_float[i]);

	for (int i = 0; i < size_wa; i++) {
		r[i] = exp(results_float[i]) / sum;
		c[i] = i;
	}
	for (int i = 0; i < size_wa; i++) {
		for (int j = i; j < size_wa; j++) {
			if (r[j] > r[i]) {
				float t = r[j];
				r[j] = r[i];
				r[i] = t;
				int tc = c[j];
				c[j] = c[i];
				c[i] = tc;
			}
		}
	}
	int top0 = 0;
	float topval = results_float[0];
	for (int i = 1; i < size_wa; i++) {
		if (results_float[i] > topval) {
			top0 = i;
			topval = results_float[i];
		}
	}
	return top0;
}
